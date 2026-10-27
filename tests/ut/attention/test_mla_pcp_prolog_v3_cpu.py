# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for PCP PrologV3; emulated kernels are not NPU evidence."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import NamedTuple
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
MLA = "vllm_ascend/attention/mla_v1.py"
BLOCK_SIZE = 128
LATENT_DIM = 4
ROPE_DIM = 2
ORDINARY_NOPE_DIM = 2
VALUE_DIM = 3
NUM_HEADS = 2
PROLOG_NUM_HEADS = NUM_HEADS + 2
Q_LORA_DIM = 3
SENTINEL = -73.0


def _load(name, scope, owner="AscendMLAImpl"):
    """Execute production methods without importing the NPU dependency graph."""
    source = ROOT / MLA
    nodes = ast.parse(source.read_text(encoding="utf-8")).body
    if owner is not None:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == owner).body
    node = next(node for node in nodes if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)
    node.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(source), "exec"), scope)
    return scope[name]


def _bind(impl, names, scope):
    for name in names:
        setattr(impl, name, MethodType(_load(name, scope), impl))


def _scatter(*, key, value, key_cache, value_cache, slot_mapping):
    """Independent logical scatter oracle, including the padding contract."""
    for row, slot in enumerate(slot_mapping.tolist()):
        if slot >= 0:
            block, offset = divmod(slot, key_cache.shape[1])
            key_cache[block, offset].copy_(key[row])
            value_cache[block, offset].copy_(value[row])


def _cache():
    return (
        torch.full((2, BLOCK_SIZE, 1, LATENT_DIM), SENTINEL),
        torch.full((2, BLOCK_SIZE, 1, ROPE_DIM), SENTINEL),
    )


@pytest.mark.parametrize("pcp_size,rank", [(2, 0), (2, 1), (4, 3)])
@pytest.mark.parametrize("num_decodes,num_prefills", [(0, 1), (2, 0), (1, 1), (0, 0)])
def test_pcp_metadata_preserves_rank_major_slots(pcp_size, rank, num_decodes, num_prefills):
    local_tokens = 4
    slots = torch.arange(pcp_size * local_tokens, dtype=torch.int32).view(pcp_size, local_tokens)
    slots[:, -1] = -1
    original = slots.clone()
    group = SimpleNamespace(world_size=pcp_size, rank_in_group=rank)
    scope = {
        "torch": torch,
        "get_pcp_group": lambda: group,
        "AscendAttentionState": SimpleNamespace(ChunkedPrefill=object()),
    }
    finalize = _load("_finalize_pcp_metadata", scope, "AscendMLAMetadataBuilder")
    builder = SimpleNamespace(pcp_size=pcp_size, pcp_rank=rank, is_pcp_decode_sharded=True)
    actual = num_decodes + num_prefills
    metadata = SimpleNamespace(
        num_actual_tokens=actual,
        num_decode_tokens=num_decodes,
        num_prefills=num_prefills,
        prefill=SimpleNamespace(),
        pcp_prolog_local_slots=None,
        pcp_prolog_global_slots=None,
    )

    finalize(builder, metadata, slots.flatten())

    assert metadata.uses_pcp_sharded_layout
    torch.testing.assert_close(metadata.pcp_prolog_local_slots, original[rank])
    torch.testing.assert_close(metadata.pcp_prolog_global_slots, original.flatten())
    torch.testing.assert_close(slots, original)
    if num_prefills:
        assert metadata.prefill.pcp_local_num_input_tokens == local_tokens


@pytest.mark.parametrize("num_slots,num_actual", [(5, 1), (4, 3)])
def test_pcp_metadata_rejects_inconsistent_token_extent(num_slots, num_actual):
    finalize = _load("_finalize_pcp_metadata", {"torch": torch}, "AscendMLAMetadataBuilder")
    builder = SimpleNamespace(pcp_size=2, is_pcp_decode_sharded=True)
    metadata = SimpleNamespace(num_actual_tokens=num_actual)
    with pytest.raises(RuntimeError, match="PCP"):
        finalize(builder, metadata, torch.arange(num_slots))


def test_replicated_decode_does_not_gain_prolog_metadata():
    finalize = _load("_finalize_pcp_metadata", {"torch": torch}, "AscendMLAMetadataBuilder")
    builder = SimpleNamespace(pcp_size=2, is_pcp_decode_sharded=False)
    metadata = SimpleNamespace(
        num_actual_tokens=1,
        num_decode_tokens=1,
        num_prefills=0,
        pcp_prolog_local_slots=None,
        pcp_prolog_global_slots=None,
    )
    finalize(builder, metadata, torch.tensor([0, -1, -1, -1]))
    assert metadata.pcp_prolog_local_slots is None
    assert metadata.pcp_prolog_global_slots is None


def _payload(token_ids):
    offsets = torch.arange(LATENT_DIM + ROPE_DIM, dtype=torch.float32)
    return token_ids[:, None] * 10 + offsets


def _ordinary_query(q_c):
    offsets = torch.arange(NUM_HEADS * (ORDINARY_NOPE_DIM + ROPE_DIM), dtype=torch.float32)
    return q_c[:, :1] + offsets


def _expanded_kv(latent):
    offsets = torch.arange(NUM_HEADS * (ORDINARY_NOPE_DIM + VALUE_DIM), dtype=torch.float32)
    return latent.reshape(latent.shape[0], LATENT_DIM)[:, :1] + offsets


def _composition_impl(group, metadata, events):
    def scatter_and_record(**kwargs):
        events.append("scatter")
        _scatter(**kwargs)

    scope = {
        "torch": torch,
        "NamedTuple": NamedTuple,
        "cdiv": lambda value, divisor: (value + divisor - 1) // divisor,
        "get_pcp_group": lambda: group,
        "wait_for_kv_layer_from_connector": lambda _: events.append("wait"),
        "notify_kv_cache_written": lambda _: events.append("notify"),
        "DeviceOperator": SimpleNamespace(reshape_and_cache=Mock(side_effect=scatter_and_record)),
    }
    for name in ("DecodeMLAPreprocessResult", "PrefillMLAPreprocessResult"):
        scope[name] = _load(name, scope, owner=None)

    def project_a(hidden):
        return (hidden[:, :1].expand(-1, Q_LORA_DIM + LATENT_DIM + ROPE_DIM).contiguous(),)

    impl = SimpleNamespace(
        layer_name="model.layers.0.self_attn.attn",
        dtype=torch.float32,
        layerwise_kv_cache_hook=None,
        use_pcp=True,
        use_dcp=False,
        is_pcp_decode_sharded=True,
        use_flashmla=False,
        enable_kv_nz=False,
        fa_quant_layer=False,
        num_heads=NUM_HEADS,
        mlapo_num_heads=PROLOG_NUM_HEADS,
        num_kv_heads=1,
        q_lora_rank=Q_LORA_DIM,
        kv_lora_rank=LATENT_DIM,
        qk_rope_head_dim=ROPE_DIM,
        qk_nope_head_dim=ORDINARY_NOPE_DIM,
        qk_head_dim=ORDINARY_NOPE_DIM + ROPE_DIM,
        v_head_dim=VALUE_DIM,
        fused_qkv_a_proj=Mock(side_effect=project_a),
        q_a_layernorm=Mock(side_effect=lambda value: value * 2),
        q_proj=Mock(side_effect=lambda value: (_ordinary_query(value),)),
        kv_b_proj=Mock(side_effect=lambda value: (_expanded_kv(value),)),
        rope_single=Mock(side_effect=lambda value, cos, sin: value + 100),
        reorg_decode_q=lambda q, pe: (q, pe),
        exec_kv_prefill=Mock(side_effect=AssertionError("PCP Prolog already wrote the current KV")),
        get_num_actual_tokens=lambda meta: meta.num_actual_tokens,
        _prolog_v3_weights={"num_heads": PROLOG_NUM_HEADS},
        _pcp_prolog_v3_buffers=(
            torch.empty(1, BLOCK_SIZE, 1, LATENT_DIM),
            torch.empty(1, BLOCK_SIZE, 1, ROPE_DIM),
            torch.arange(BLOCK_SIZE),
        ),
    )

    def run(hidden, cos, sin, cache, indices, *, quantize_query):
        events.append("prolog")
        assert not quantize_query
        torch.testing.assert_close(indices, torch.arange(hidden.shape[0]))
        payload = _payload(hidden[:, 0])
        # Padding outputs are deliberately non-finite. The transport must
        # replace invalid rows with zero, not multiply NaN/Inf by zero.
        invalid = metadata.pcp_prolog_local_slots < 0
        payload[invalid, 0] = torch.nan
        payload[invalid, -1] = torch.inf
        _scatter(
            key=payload[:, :LATENT_DIM].unsqueeze(1),
            value=payload[:, LATENT_DIM:].unsqueeze(1),
            key_cache=cache[0],
            value_cache=cache[1],
            slot_mapping=indices,
        )
        # Distinct token/head/feature values expose wrong decode slicing,
        # including accidental retention of the operator's padded heads.
        tokens = hidden[:, :1].unsqueeze(-1)
        heads = torch.arange(PROLOG_NUM_HEADS)[None, :, None] * 10
        q = tokens * 100 + heads + torch.arange(LATENT_DIM)
        pe = tokens * 1000 + heads + torch.arange(ROPE_DIM)
        return q, pe, None, None, None

    impl._run_prolog_v3 = Mock(side_effect=run)
    _bind(
        impl,
        ("mla_preprocess_pcp_prolog_v3", "_pcp_prolog_v3_prefill_result"),
        scope,
    )
    return impl, scope


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("kind", ["mixed", "prefill", "decode", "empty_local"])
@pytest.mark.parametrize("layerwise", [False, True])
def test_pcp_prolog_composition_preserves_cache_and_ordinary_prefill_query(rank, kind, layerwise):
    slots = torch.tensor([[0, 129, 2, -1], [5, 128, -1, -1]])
    if kind == "decode":
        slots[0, 2] = -1
    if kind == "empty_local":
        slots[1].fill_(-1)
    token_ids = torch.tensor([[1.0, 2.0, 3.0, 4.0], [11.0, 12.0, 13.0, 14.0]])
    actual = int((slots[rank] >= 0).sum())
    decode_tokens = actual if kind == "decode" else (1 if kind == "mixed" else 0)
    local_tokens = slots.shape[1]
    prefill_tokens = actual - decode_tokens
    metadata = SimpleNamespace(
        num_actual_tokens=actual,
        num_decode_tokens=decode_tokens,
        num_decodes=int(decode_tokens > 0),
        num_prefills=int(prefill_tokens > 0),
        slot_mapping=slots.flatten(),
        uses_pcp_sharded_layout=True,
        pcp_prolog_local_slots=slots[rank],
        pcp_prolog_global_slots=slots.flatten(),
        pcp_cos=torch.ones(local_tokens, 1, 1, ROPE_DIM),
        pcp_sin=torch.zeros(local_tokens, 1, 1, ROPE_DIM),
        prefill=SimpleNamespace(
            cos=torch.ones(local_tokens - decode_tokens, 1, 1, ROPE_DIM),
            sin=torch.zeros(local_tokens - decode_tokens, 1, 1, ROPE_DIM),
            pcp_local_num_input_tokens=local_tokens,
        ),
    )
    gathered = torch.cat([_payload(ids) for ids in token_ids])
    gathered[slots.flatten() < 0] = 0
    events = []

    def gather(packed, dim):
        events.append("gather")
        assert dim == 0
        torch.testing.assert_close(packed, gathered.view(2, local_tokens, -1)[rank])
        assert torch.isfinite(packed).all()
        return gathered.clone()

    group = SimpleNamespace(world_size=2, rank_in_group=rank, all_gather=Mock(side_effect=gather))
    impl, scope = _composition_impl(group, metadata, events)
    if layerwise:
        impl.layerwise_kv_cache_hook = SimpleNamespace(
            layerwise=True, wait_for_layer=lambda _: events.append("layer_wait")
        )
    cache = _cache()
    expected = tuple(tensor.clone() for tensor in cache)
    _scatter(
        key=gathered[:, :LATENT_DIM].unsqueeze(1),
        value=gathered[:, LATENT_DIM:].unsqueeze(1),
        key_cache=expected[0],
        value_cache=expected[1],
        slot_mapping=slots.flatten(),
    )

    decode, prefill = impl.mla_preprocess_pcp_prolog_v3(token_ids[rank, :, None], cache, metadata)

    expected_events = ["wait"] if prefill_tokens or layerwise else []
    if layerwise:
        expected_events.append("layer_wait")
    expected_events += ["prolog", "gather", "scatter", "notify"]
    assert events == expected_events
    group.all_gather.assert_called_once()
    impl._run_prolog_v3.assert_called_once()
    scope["DeviceOperator"].reshape_and_cache.assert_called_once()
    for written, reference in zip(cache, expected):
        torch.testing.assert_close(written, reference)
    assert (decode is not None) == bool(decode_tokens)
    assert (prefill is not None) == bool(prefill_tokens)
    if decode is not None:
        expected_q = torch.empty(decode_tokens, NUM_HEADS, LATENT_DIM)
        expected_pe = torch.empty(decode_tokens, NUM_HEADS, ROPE_DIM)
        for row, token_id in enumerate(token_ids[rank, :decode_tokens]):
            for head in range(NUM_HEADS):
                expected_q[row, head] = token_id * 100 + head * 10 + torch.arange(LATENT_DIM)
                expected_pe[row, head] = token_id * 1000 + head * 10 + torch.arange(ROPE_DIM)
        torch.testing.assert_close(decode.ql_nope, expected_q)
        torch.testing.assert_close(decode.q_pe, expected_pe)
        assert decode.k_nope is cache[0] and decode.k_pe is cache[1]
    if prefill is not None:
        local_ids = token_ids[rank, decode_tokens:actual]
        q_c = local_ids[:, None].expand(-1, Q_LORA_DIM) * 2
        ordinary_q = _ordinary_query(q_c).view(prefill_tokens, NUM_HEADS, ORDINARY_NOPE_DIM + ROPE_DIM)
        torch.testing.assert_close(prefill.q_nope, ordinary_q[..., :ORDINARY_NOPE_DIM])
        torch.testing.assert_close(prefill.q_pe, ordinary_q[..., ORDINARY_NOPE_DIM:] + 100)
        payload = _payload(local_ids)
        expanded = _expanded_kv(payload[:, :LATENT_DIM]).view(prefill_tokens, NUM_HEADS, ORDINARY_NOPE_DIM + VALUE_DIM)
        torch.testing.assert_close(prefill.k_nope, expanded[..., :ORDINARY_NOPE_DIM])
        torch.testing.assert_close(prefill.value, expanded[..., ORDINARY_NOPE_DIM:])
        torch.testing.assert_close(prefill.k_pe, payload[:, None, LATENT_DIM:].expand(-1, NUM_HEADS, -1))
        impl.q_proj.assert_called_once()
    else:
        impl.q_proj.assert_not_called()
    impl.exec_kv_prefill.assert_not_called()


def test_zero_token_pcp_step_skips_prolog_and_collectives():
    group = SimpleNamespace(world_size=2, rank_in_group=0, all_gather=Mock())
    metadata = SimpleNamespace(
        num_actual_tokens=0,
        num_decode_tokens=0,
        num_decodes=0,
        num_prefills=0,
        pcp_prolog_local_slots=torch.empty(0, dtype=torch.int64),
        pcp_prolog_global_slots=torch.empty(0, dtype=torch.int64),
        pcp_cos=torch.empty(0, ROPE_DIM),
        pcp_sin=torch.empty(0, ROPE_DIM),
    )
    impl, scope = _composition_impl(group, metadata, [])
    assert impl.mla_preprocess_pcp_prolog_v3(torch.empty(0, 1), _cache(), metadata) == (None, None)
    impl._run_prolog_v3.assert_not_called()
    group.all_gather.assert_not_called()
    scope["DeviceOperator"].reshape_and_cache.assert_not_called()


def _route_impl():
    class NoopOffloader:
        pass

    scope = {
        "torch": torch,
        "torch_npu": SimpleNamespace(npu_mla_prolog_v3=object()),
        "NoopOffloader": NoopOffloader,
        "get_offloader": lambda: NoopOffloader(),
        "PROLOG_V3_SUPPORTED_INPUT_SIZES": (7168,),
        "AscendW8A8LinearMethod": type("StaticQuant", (), {}),
        "AscendW8A8DynamicLinearMethod": type("DynamicQuant", (), {}),
        "AscendW8A8MXFP8DynamicLinearMethod": type("MXFPQuant", (), {}),
    }
    impl = SimpleNamespace(
        vllm_config=SimpleNamespace(use_v2_model_runner=True),
        use_pcp=True,
        use_dcp=False,
        is_pcp_decode_sharded=True,
        num_kv_heads=1,
        use_mla_rope=True,
        qk_rope_head_dim=ROPE_DIM,
        kv_lora_rank=LATENT_DIM,
        dtype=torch.bfloat16,
        fa_quant_layer=False,
        enable_kv_nz=False,
        fused_qkv_a_proj=SimpleNamespace(input_size=7168),
        q_proj=SimpleNamespace(_chunk_size=0),
        q_lora_rank=Q_LORA_DIM,
        q_a_layernorm=object(),
        kv_a_layernorm=object(),
        _pcp_prolog_v3_buffers=object(),
        _prolog_v3_weights={"num_heads": NUM_HEADS},
    )
    impl._get_context_prolog_quant_method = _load("_get_context_prolog_quant_method", scope)
    _bind(
        impl,
        (
            "supports_prolog_v3_dtype",
            "supports_prolog_v3_projection",
            "supports_prolog_v3_quantization",
            "_supports_strided_mla_projection",
            "supports_prolog_v3_writer",
            "supports_pcp_prolog_v3",
            "_can_use_pcp_prolog_v3",
        ),
        scope,
    )
    return impl, scope


@pytest.mark.parametrize(
    "unsupported",
    [
        None,
        "non_pcp",
        "dcp",
        "replicated_decode",
        "fp16",
        "quantized_cache",
        "nz_cache",
        "quantized_weights",
        "mixed_weights",
        "missing_projection",
        "output_gate",
        "chunked_q",
        "runner_v1",
        "offloader",
    ],
)
def test_pcp_prolog_admission_keeps_unsupported_routes_on_native(unsupported):
    impl, scope = _route_impl()
    if unsupported == "non_pcp":
        impl.use_pcp = False
    elif unsupported == "dcp":
        impl.use_dcp = True
    elif unsupported == "replicated_decode":
        impl.is_pcp_decode_sharded = False
    elif unsupported == "fp16":
        impl.dtype = torch.float16
    elif unsupported == "quantized_cache":
        impl.fa_quant_layer = True
    elif unsupported == "nz_cache":
        impl.enable_kv_nz = True
    elif unsupported in ("quantized_weights", "mixed_weights"):
        quant = scope["AscendW8A8DynamicLinearMethod"]()
        impl.fused_qkv_a_proj.quant_method = SimpleNamespace(quant_method=quant)
        if unsupported == "quantized_weights":
            impl.q_proj.quant_method = SimpleNamespace(quant_method=quant)
    elif unsupported == "missing_projection":
        impl.fused_qkv_a_proj = None
    elif unsupported == "output_gate":
        impl.qkv_a_gate_fused = True
    elif unsupported == "chunked_q":
        impl.q_proj._chunk_size = 128
    elif unsupported == "runner_v1":
        impl.vllm_config.use_v2_model_runner = False
    elif unsupported == "offloader":
        scope["get_offloader"] = object
    assert impl.supports_pcp_prolog_v3() is (unsupported is None)


@pytest.mark.parametrize(
    "unsupported",
    [None, "unpartitioned", "flashmla", "no_workspace", "no_weights", "packed", "strided", "fp16", "rope_shape"],
)
def test_pcp_prolog_batch_route_requires_matching_dense_cache_and_layout(unsupported):
    impl, _ = _route_impl()
    cache = tuple(component.to(torch.bfloat16) for component in _cache())
    metadata = SimpleNamespace(uses_pcp_sharded_layout=True, external_flashmla=None)
    if unsupported == "unpartitioned":
        metadata.uses_pcp_sharded_layout = False
    elif unsupported == "flashmla":
        metadata.external_flashmla = object()
    elif unsupported == "no_workspace":
        impl._pcp_prolog_v3_buffers = None
    elif unsupported == "no_weights":
        impl._prolog_v3_weights = None
    elif unsupported == "packed":
        cache = (cache[0],)
    elif unsupported == "strided":
        fused = torch.zeros(2, BLOCK_SIZE, 1, LATENT_DIM + ROPE_DIM, dtype=torch.bfloat16)
        cache = (fused[..., :LATENT_DIM], fused[..., LATENT_DIM:])
    elif unsupported == "fp16":
        cache = tuple(component.to(torch.float16) for component in cache)
    elif unsupported == "rope_shape":
        cache = (cache[0], cache[1][:1])
    assert impl._can_use_pcp_prolog_v3(cache, metadata) is (unsupported is None)


def test_pcp_prolog_workspace_covers_graph_padding_and_keeps_addresses():
    scope = {"torch": torch, "cdiv": lambda value, divisor: (value + divisor - 1) // divisor}
    prepare = _load("_prepare_pcp_prolog_v3_buffers", scope)
    capacity = BLOCK_SIZE + 3
    impl = SimpleNamespace(
        _pcp_prolog_v3_buffers=None,
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[8, capacity]),
            cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
        ),
        num_kv_heads=1,
        kv_lora_rank=LATENT_DIM,
        qk_rope_head_dim=ROPE_DIM,
        dtype=torch.bfloat16,
        W_UV=torch.empty(1),
    )
    prepare(impl)
    buffers = impl._pcp_prolog_v3_buffers
    assert buffers[0].shape == (2, BLOCK_SIZE, 1, LATENT_DIM)
    assert buffers[1].shape == (2, BLOCK_SIZE, 1, ROPE_DIM)
    torch.testing.assert_close(buffers[2], torch.arange(capacity, dtype=torch.int64))
    addresses = tuple(tensor.data_ptr() for tensor in buffers)
    prepare(impl)
    assert impl._pcp_prolog_v3_buffers is buffers
    assert tuple(tensor.data_ptr() for tensor in buffers) == addresses


@pytest.mark.parametrize(
    "invalid", ["missing_slots", "missing_rope", "row_count", "global_count", "actual", "workspace"]
)
def test_pcp_prolog_rejects_invalid_contract_before_collectives(invalid):
    group = SimpleNamespace(world_size=2, rank_in_group=0, all_gather=Mock())
    metadata = SimpleNamespace(
        num_actual_tokens=1,
        num_decode_tokens=1,
        num_decodes=1,
        num_prefills=0,
        pcp_prolog_local_slots=torch.tensor([0, -1]),
        pcp_prolog_global_slots=torch.tensor([0, -1, 1, -1]),
        pcp_cos=torch.ones(2, ROPE_DIM),
        pcp_sin=torch.zeros(2, ROPE_DIM),
    )
    events = []
    impl, scope = _composition_impl(group, metadata, events)
    if invalid == "missing_slots":
        metadata.pcp_prolog_local_slots = None
    elif invalid == "missing_rope":
        metadata.pcp_cos = None
    elif invalid == "row_count":
        metadata.pcp_cos = metadata.pcp_cos[:1]
    elif invalid == "global_count":
        metadata.pcp_prolog_global_slots = metadata.pcp_prolog_global_slots[:2]
    elif invalid == "actual":
        metadata.num_actual_tokens = 3
    elif invalid == "workspace":
        nope, rope, indices = impl._pcp_prolog_v3_buffers
        impl._pcp_prolog_v3_buffers = (nope, rope, indices[:1])
    with pytest.raises(RuntimeError, match="PCP PROLOG_V3"):
        impl.mla_preprocess_pcp_prolog_v3(torch.ones(2, 1), _cache(), metadata)
    assert events == []
    impl._run_prolog_v3.assert_not_called()
    group.all_gather.assert_not_called()
    scope["DeviceOperator"].reshape_and_cache.assert_not_called()
