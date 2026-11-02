# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCP preprocessing orchestration with mocked kernels and communication."""

from concurrent.futures import ThreadPoolExecutor
from threading import Condition, local
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch_npu
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.offloader.base import NoopOffloader

from vllm_ascend.attention import mla_v1
from vllm_ascend.attention.context_parallel import mla_pcp
from vllm_ascend.attention.context_parallel.mla_cp import AscendMlaDCPImpl
from vllm_ascend.attention.context_parallel.mla_pcp import AscendMlaPCPImpl
from vllm_ascend.attention.mla_v1 import AscendMLABackend, AscendMLAImpl
from vllm_ascend.quantization.methods import AscendW8A8DynamicLinearMethod

BLOCK_SIZE = 128
INPUT_SIZE = 7168
LATENT_DIM = 4
ROPE_DIM = 2
NOPE_DIM = 2
VALUE_DIM = 3
NUM_HEADS = 2
PROLOG_HEADS = 4
Q_LORA_DIM = 3
SENTINEL = -73.0


def _scatter(*, key, value, key_cache, value_cache, slot_mapping):
    for row, slot in enumerate(slot_mapping.tolist()):
        if slot >= 0:
            block, offset = divmod(slot, key_cache.shape[1])
            key_cache[block, offset].copy_(key[row])
            value_cache[block, offset].copy_(value[row])


def _cache(dtype=torch.bfloat16):
    return (
        torch.full((4, BLOCK_SIZE, 1, LATENT_DIM), SENTINEL, dtype=dtype),
        torch.full((4, BLOCK_SIZE, 1, ROPE_DIM), SENTINEL, dtype=dtype),
    )


@pytest.fixture
def pcp_runtime(monkeypatch):
    # Module patches are installed once, outside the worker threads. Each rank
    # retains its own communicator and event log, as separate processes would.
    rank_state = local()
    native_preprocess = Mock(return_value=("native-decode", "native-prefill", "native-gate"))
    monkeypatch.setattr(AscendMLAImpl, "_mla_preprocess", native_preprocess)
    monkeypatch.setattr(mla_pcp, "get_pcp_group", lambda: rank_state.group)
    monkeypatch.setattr(mla_pcp, "wait_for_kv_layer_from_connector", lambda _: rank_state.events.append("wait"))
    monkeypatch.setattr(mla_pcp, "notify_kv_cache_written", lambda _: rank_state.events.append("notify"))

    def scatter(**kwargs):
        rank_state.events.append("scatter")
        _scatter(**kwargs)

    monkeypatch.setattr(mla_pcp.DeviceOperator, "reshape_and_cache", scatter)
    for module in (mla_pcp, mla_v1):
        monkeypatch.setattr(module, "get_offloader", lambda: NoopOffloader())
        monkeypatch.setattr(module, "enable_sfa", lambda _: False)
    monkeypatch.setattr(
        torch_npu,
        "npu_mla_prolog_v3",
        Mock(side_effect=AssertionError("unit tests must mock the NPU kernel")),
        raising=False,
    )
    return SimpleNamespace(rank_state=rank_state, native_preprocess=native_preprocess)


def _impl(runtime, group=None, events=None):
    runtime.rank_state.group = group
    runtime.rank_state.events = [] if events is None else events
    config = SimpleNamespace(
        use_v2_model_runner=True,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=BLOCK_SIZE + 3),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[4, BLOCK_SIZE + 7], static_forward_context={}),
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
    )
    # Follow the neighbouring MLA unit tests: initialise only the state needed
    # by preprocessing, without constructing distributed linear layers.
    impl = AscendMlaPCPImpl.__new__(AscendMlaPCPImpl)
    impl.__dict__.update(
        vllm_config=config,
        use_pcp=True,
        use_dcp=False,
        is_pcp_decode_sharded=True,
        use_flashmla=False,
        use_mla_rope=True,
        enable_kv_nz=False,
        enable_mlapo=False,
        fa_quant_layer=False,
        support_fp8_attention=False,
        is_draft_model=False,
        dtype=torch.bfloat16,
        num_heads=NUM_HEADS,
        mlapo_num_heads=PROLOG_HEADS,
        num_kv_heads=1,
        q_lora_rank=Q_LORA_DIM,
        kv_lora_rank=LATENT_DIM,
        qk_rope_head_dim=ROPE_DIM,
        qk_nope_head_dim=NOPE_DIM,
        qk_head_dim=NOPE_DIM + ROPE_DIM,
        v_head_dim=VALUE_DIM,
        layer_name="model.layers.0.self_attn.attn",
        layerwise_kv_cache_hook=None,
        W_UV=torch.empty(NUM_HEADS, LATENT_DIM, VALUE_DIM),
        native_preprocess=runtime.native_preprocess,
        _pcp_prolog_v3_buffers=None,
    )
    impl.fused_qkv_a_proj = Mock(
        side_effect=lambda hidden: (hidden[:, :1].expand(-1, Q_LORA_DIM + LATENT_DIM + ROPE_DIM).contiguous(),)
    )
    impl.fused_qkv_a_proj.input_size = INPUT_SIZE
    impl.fused_qkv_a_proj.quant_method = UnquantizedLinearMethod()
    impl.q_a_layernorm = Mock(side_effect=lambda value: value * 2)
    impl.kv_a_layernorm = object()
    impl.q_proj = Mock(side_effect=lambda value: (value[:, :1] + torch.arange(NUM_HEADS * (NOPE_DIM + ROPE_DIM)),))
    impl.q_proj.quant_method = UnquantizedLinearMethod()
    impl.q_proj._chunk_size = 0
    impl.qkv_a_gate_fused = False
    impl.kv_b_proj = Mock(
        side_effect=lambda value: (value[:, :1].float() + torch.arange(NUM_HEADS * (NOPE_DIM + VALUE_DIM)),)
    )
    impl.rope_single = Mock(side_effect=lambda value, cos, sin: value + 100)
    impl.exec_kv_prefill = Mock(side_effect=AssertionError("current KV must be published exactly once"))
    impl._prolog_v3_weights = {"num_heads": PROLOG_HEADS}
    impl._prepare_pcp_prolog_v3_buffers()
    return impl


def _metadata(slots, rank, actual, decodes):
    local_tokens = slots.shape[1]
    return SimpleNamespace(
        num_actual_tokens=actual,
        num_decode_tokens=decodes,
        num_decodes=int(decodes > 0),
        num_prefills=int(actual > decodes),
        slot_mapping=slots.flatten(),
        uses_pcp_sharded_layout=True,
        external_flashmla=None,
        pcp_cos=torch.ones(local_tokens, 1, 1, ROPE_DIM),
        pcp_sin=torch.zeros(local_tokens, 1, 1, ROPE_DIM),
        prefill=SimpleNamespace(
            cos=torch.ones(local_tokens - decodes, 1, 1, ROPE_DIM),
            sin=torch.zeros(local_tokens - decodes, 1, 1, ROPE_DIM),
            pcp_local_num_input_tokens=local_tokens,
        ),
    )


def _payload(token_ids):
    return (token_ids[:, None] * 10 + torch.arange(LATENT_DIM + ROPE_DIM)).to(torch.bfloat16)


def _emulate_prolog(impl, metadata, rank, group, events):
    def run(hidden, cos, sin, cache, indices, *, quantize_query):
        events.append("prolog")
        assert not quantize_query
        torch.testing.assert_close(indices, torch.arange(hidden.shape[0]))
        payload = _payload(hidden[:, 0])
        local_slots = metadata.slot_mapping.view(group.world_size, -1)[rank]
        payload[local_slots < 0, 0] = torch.nan
        payload[local_slots < 0, -1] = torch.inf
        _scatter(
            key=payload[:, :LATENT_DIM].unsqueeze(1),
            value=payload[:, LATENT_DIM:].unsqueeze(1),
            key_cache=cache[0],
            value_cache=cache[1],
            slot_mapping=indices,
        )
        tokens = hidden[:, :1].unsqueeze(-1)
        heads = torch.arange(PROLOG_HEADS)[None, :, None] * 10
        return tokens * 100 + heads + torch.arange(LATENT_DIM), tokens * 1000 + heads + torch.arange(ROPE_DIM), None

    impl._run_prolog_v3 = Mock(side_effect=run)


class RendezvousGather:
    """Gather submitted rank payloads, rather than return a precooked oracle."""

    def __init__(self, size):
        self.size = size
        self.condition = Condition()
        self.payloads = {}

    def gather(self, rank, tensor, dim):
        assert dim == 0
        with self.condition:
            assert rank not in self.payloads, "unexpected second KV collective"
            self.payloads[rank] = tensor.clone()
            self.condition.notify_all()
            assert self.condition.wait_for(lambda: len(self.payloads) == self.size, timeout=10), "missing PCP rank"
            assert len({tuple(value.shape) for value in self.payloads.values()}) == 1
            return torch.cat([self.payloads[index] for index in range(self.size)])


@pytest.mark.parametrize("size", [2, 4])
@pytest.mark.parametrize("kind", ["decode", "prefill", "mixed"])
@pytest.mark.parametrize("hook_kind", ["none", "ordinary", "layerwise"])
def test_all_ranks_publish_once_and_native_prefill_consumes_written_kv(pcp_runtime, size, kind, hook_kind):
    actuals = [3, 2] + [1, 0] * (size // 4)
    if kind == "decode":
        actuals[-1] = 0
        decodes = actuals.copy()
    elif kind == "prefill":
        decodes = [0] * size
    else:
        decodes = [1, 0] + [1, 0] * (size // 4)
    slots = torch.full((size, 4), -1, dtype=torch.int64)
    for rank, actual in enumerate(actuals):
        slots[rank, :actual] = torch.tensor([rank * 11, 129 + rank * 11, 260 + rank * 11])[:actual]
    original_slots = slots.clone()
    token_ids = torch.arange(1, size * 4 + 1, dtype=torch.float32).view(size, 4)
    rendezvous = RendezvousGather(size)

    def execute(rank):
        events = []

        def gather(tensor, dim):
            events.append("gather")
            return rendezvous.gather(rank, tensor, dim)

        group = SimpleNamespace(world_size=size, rank_in_group=rank, all_gather=Mock(side_effect=gather))
        impl = _impl(pcp_runtime, group, events)
        metadata = _metadata(slots, rank, actuals[rank], decodes[rank])
        if hook_kind != "none":
            impl.layerwise_kv_cache_hook = SimpleNamespace(
                layerwise=hook_kind == "layerwise", wait_for_layer=Mock(side_effect=lambda _: events.append("hook"))
            )
        _emulate_prolog(impl, metadata, rank, group, events)
        cache = _cache()
        result = impl._mla_preprocess(impl.layer_name, token_ids[rank, :, None], cache, metadata)
        return impl, result, cache, events

    with ThreadPoolExecutor(max_workers=size) as pool:
        results = list(pool.map(execute, range(size)))
    expected_payload = _payload(token_ids.flatten())
    expected_payload[slots.flatten() < 0] = 0
    expected_cache = _cache()
    _scatter(
        key=expected_payload[:, :LATENT_DIM].unsqueeze(1),
        value=expected_payload[:, LATENT_DIM:].unsqueeze(1),
        key_cache=expected_cache[0],
        value_cache=expected_cache[1],
        slot_mapping=slots.flatten(),
    )
    for rank, (impl, (decode, prefill, gate), cache, events) in enumerate(results):
        actual, count = actuals[rank], decodes[rank]
        torch.testing.assert_close(cache, expected_cache)
        assert torch.isfinite(rendezvous.payloads[rank]).all()
        impl.native_preprocess.assert_not_called()
        impl.exec_kv_prefill.assert_not_called()
        assert gate is None
        expected_events = ["wait"] if actual > count or hook_kind == "layerwise" else []
        if hook_kind != "none":
            expected_events.append("hook")
        assert events == expected_events + ["prolog", "gather", "scatter", "notify"]
        if count:
            expected_q = token_ids[rank, :count, None, None] * 100
            expected_q = expected_q + torch.arange(NUM_HEADS)[None, :, None] * 10 + torch.arange(LATENT_DIM)
            torch.testing.assert_close(decode.ql_nope, expected_q)
            assert decode.k_nope is cache[0]
        else:
            assert decode is None
        if actual > count:
            ordinary = token_ids[rank, count:actual, None] * 2 + torch.arange(NUM_HEADS * (NOPE_DIM + ROPE_DIM))
            ordinary = ordinary.view(-1, NUM_HEADS, NOPE_DIM + ROPE_DIM)
            torch.testing.assert_close(prefill.q_nope, ordinary[..., :NOPE_DIM])
            torch.testing.assert_close(prefill.q_pe, ordinary[..., NOPE_DIM:] + 100)
            local = _payload(token_ids[rank, count:actual])
            expanded = (local[:, :1].float() + torch.arange(NUM_HEADS * (NOPE_DIM + VALUE_DIM))).view(
                -1, NUM_HEADS, NOPE_DIM + VALUE_DIM
            )
            torch.testing.assert_close(prefill.k_nope, expanded[..., :NOPE_DIM])
            torch.testing.assert_close(prefill.value, expanded[..., NOPE_DIM:])
            torch.testing.assert_close(prefill.k_pe, local[:, None, LATENT_DIM:].expand(-1, NUM_HEADS, -1))
            assert impl.fused_qkv_a_proj.call_args.args[0].shape[0] == actual
        else:
            assert prefill is None
            impl.fused_qkv_a_proj.assert_not_called()
    torch.testing.assert_close(slots, original_slots)


@pytest.mark.parametrize(
    "unsupported",
    [
        "none",
        "non_pcp",
        "dcp",
        "replicated",
        "fp16",
        "quantized_cache",
        "nz",
        "quantized_weights",
        "unknown_wrapper",
        "mixed_weights",
        "missing_projection",
        "fused_gate",
        "chunked_q",
        "runner_v1",
        "offloader",
        "sfa",
        "flashmla",
        "input_width",
        "no_rope",
        "multiple_kv_heads",
    ],
)
def test_layer_admission_rejects_unsupported_projections_and_topologies(pcp_runtime, monkeypatch, unsupported):
    impl = _impl(pcp_runtime)
    fields = {
        "non_pcp": ("use_pcp", False),
        "dcp": ("use_dcp", True),
        "replicated": ("is_pcp_decode_sharded", False),
        "fp16": ("dtype", torch.float16),
        "quantized_cache": ("fa_quant_layer", True),
        "nz": ("enable_kv_nz", True),
        "missing_projection": ("fused_qkv_a_proj", None),
        "fused_gate": ("qkv_a_gate_fused", True),
        "flashmla": ("use_flashmla", True),
        "no_rope": ("use_mla_rope", False),
        "multiple_kv_heads": ("num_kv_heads", 2),
    }
    if unsupported in fields:
        setattr(impl, *fields[unsupported])
    elif unsupported in ("quantized_weights", "mixed_weights", "unknown_wrapper"):
        method = SimpleNamespace()
        if unsupported != "unknown_wrapper":
            method.quant_method = object.__new__(AscendW8A8DynamicLinearMethod)
        impl.fused_qkv_a_proj.quant_method = method
        if unsupported != "mixed_weights":
            impl.q_proj.quant_method = method
    elif unsupported == "chunked_q":
        impl.q_proj._chunk_size = 128
    elif unsupported == "runner_v1":
        impl.vllm_config.use_v2_model_runner = False
    elif unsupported == "offloader":
        monkeypatch.setattr(mla_pcp, "get_offloader", object)
    elif unsupported == "sfa":
        monkeypatch.setattr(mla_pcp, "enable_sfa", lambda _: True)
    elif unsupported == "input_width":
        impl.fused_qkv_a_proj.input_size = 7
    assert impl.supports_pcp_prolog_v3() is (unsupported == "none")


@pytest.mark.parametrize(
    "unsupported", ["layout", "flash", "workspace", "weights", "packed", "strided", "dtype", "shape"]
)
def test_runtime_fallback_preserves_super_return_contract(pcp_runtime, unsupported):
    impl = _impl(pcp_runtime)
    cache = _cache()
    metadata = _metadata(torch.tensor([[0, -1], [1, -1]]), 0, 1, 1)
    if unsupported == "layout":
        metadata.uses_pcp_sharded_layout = False
    elif unsupported == "flash":
        metadata.external_flashmla = object()
    elif unsupported == "workspace":
        impl._pcp_prolog_v3_buffers = None
    elif unsupported == "weights":
        impl._prolog_v3_weights = None
    elif unsupported == "packed":
        cache = cache[:1]
    elif unsupported == "strided":
        fused = torch.zeros(4, BLOCK_SIZE, 1, LATENT_DIM + ROPE_DIM, dtype=torch.bfloat16)
        cache = fused[..., :LATENT_DIM], fused[..., LATENT_DIM:]
    elif unsupported == "dtype":
        cache = _cache(torch.float16)
    elif unsupported == "shape":
        cache = cache[0], cache[1][:1]
    hidden = torch.ones(2, 1)
    assert impl._mla_preprocess(impl.layer_name, hidden, cache, metadata) == impl.native_preprocess.return_value
    impl.native_preprocess.assert_called_once_with(impl.layer_name, hidden, cache, metadata)


@pytest.mark.parametrize("invalid", ["rope", "rows", "slots", "actual", "decodes", "capacity"])
def test_invalid_row_contract_fails_before_kernel_or_collective(pcp_runtime, invalid):
    events = []
    group = SimpleNamespace(world_size=2, rank_in_group=0, all_gather=Mock())
    impl = _impl(pcp_runtime, group, events)
    metadata = _metadata(torch.tensor([[0, -1], [1, -1]]), 0, 1, 1)
    impl._run_prolog_v3 = Mock()
    if invalid == "rope":
        metadata.pcp_cos = None
    elif invalid == "rows":
        metadata.pcp_sin = metadata.pcp_sin[:1]
    elif invalid == "slots":
        metadata.slot_mapping = metadata.slot_mapping[:2]
    elif invalid == "actual":
        metadata.num_actual_tokens = 3
    elif invalid == "decodes":
        metadata.num_decode_tokens = 2
    elif invalid == "capacity":
        nope, rope, indices = impl._pcp_prolog_v3_buffers
        impl._pcp_prolog_v3_buffers = nope, rope, indices[:1]
    with pytest.raises(RuntimeError, match="PCP PROLOG_V3"):
        impl._mla_preprocess(impl.layer_name, torch.ones(2, 1), _cache(), metadata)
    assert events == []
    impl._run_prolog_v3.assert_not_called()
    group.all_gather.assert_not_called()


def test_zero_global_extent_does_not_launch_collective(pcp_runtime):
    group = SimpleNamespace(world_size=2, rank_in_group=0, all_gather=Mock())
    impl = _impl(pcp_runtime, group)
    metadata = _metadata(torch.empty(2, 0, dtype=torch.int64), 0, 0, 0)
    assert impl._mla_preprocess(impl.layer_name, torch.empty(0, 1), _cache(), metadata) == (None, None, None)
    group.all_gather.assert_not_called()


@pytest.mark.parametrize("supported", [True, False])
def test_actual_base_loader_refreshes_private_weights_without_reallocating_workspace(
    pcp_runtime, monkeypatch, supported
):
    impl = _impl(pcp_runtime)
    impl._prolog_v3_weights = None
    impl._pcp_prolog_v3_buffers = None
    impl.dtype = torch.bfloat16 if supported else torch.float16
    impl.kv_b_proj.quant_method = UnquantizedLinearMethod()
    impl.kv_b_proj.weight = torch.ones(NUM_HEADS * (NOPE_DIM + VALUE_DIM), LATENT_DIM, dtype=impl.dtype)
    impl.fused_qkv_a_proj.weight = torch.ones(Q_LORA_DIM + LATENT_DIM + ROPE_DIM, INPUT_SIZE, dtype=impl.dtype)
    impl.q_proj.weight = torch.ones(NUM_HEADS * (NOPE_DIM + ROPE_DIM), Q_LORA_DIM, dtype=impl.dtype)
    monkeypatch.setattr(torch_npu, "npu_format_cast", lambda tensor, _: tensor.clone())
    monkeypatch.setattr(mla_v1, "maybe_trans_nz", lambda tensor: tensor)
    # Exercise the real builder and parameter replacement as well as the
    # inherited loading hook; only the device format conversion is emulated.
    impl._build_prolog_v3_weights = Mock(wraps=impl._build_prolog_v3_weights)
    impl.process_weights_after_loading(impl.dtype)
    if not supported:
        assert impl._prolog_v3_weights is None
        assert impl._pcp_prolog_v3_buffers is None
        impl._build_prolog_v3_weights.assert_not_called()
        return
    weights = impl._prolog_v3_weights
    buffers = impl._pcp_prolog_v3_buffers
    assert buffers[0].shape == (2, BLOCK_SIZE, 1, LATENT_DIM)
    assert buffers[2].numel() == BLOCK_SIZE + 7
    pointers = tuple(tensor.data_ptr() for tensor in buffers)
    weight_pointer = weights["weight_uk"].data_ptr()
    projection_names = ("weight_dq", "weight_dkv_kr", "weight_uq_qr")
    projection_pointers = {name: weights[name].data_ptr() for name in projection_names}
    impl.kv_b_proj.weight.fill_(9)
    impl.fused_qkv_a_proj.weight.fill_(5)
    impl.q_proj.weight.fill_(7)
    # These format-converted weights own storage. Unlike weight_uk, they must
    # be refreshed explicitly rather than following an aliased source tensor.
    for name in projection_names:
        torch.testing.assert_close(weights[name], torch.ones_like(weights[name]))
    impl.process_weights_after_loading(impl.dtype)
    assert impl._prolog_v3_weights is weights
    assert weights["weight_uk"].data_ptr() == weight_pointer
    torch.testing.assert_close(weights["weight_uk"], torch.full_like(weights["weight_uk"], 9))
    for name, value in (("weight_dq", 5), ("weight_dkv_kr", 5), ("weight_uq_qr", 7)):
        assert weights[name].data_ptr() == projection_pointers[name]
        torch.testing.assert_close(weights[name], torch.full_like(weights[name], value))
    assert impl._pcp_prolog_v3_buffers is buffers
    assert tuple(tensor.data_ptr() for tensor in buffers) == pointers
    assert impl._build_prolog_v3_weights.call_count == 2


@pytest.mark.parametrize(
    "pcp,dcp,expected", [(1, False, "base"), (2, False, "pcp"), (1, True, "dcp"), (2, True, "dcp")]
)
def test_backend_factory_keeps_dcp_precedence(monkeypatch, pcp, dcp, expected):
    monkeypatch.setattr(mla_v1, "is_dcp_enabled", lambda: dcp)
    monkeypatch.setattr(
        mla_v1,
        "get_current_vllm_config",
        lambda: SimpleNamespace(parallel_config=SimpleNamespace(prefill_context_parallel_size=pcp)),
    )
    expected_cls = {"base": AscendMLAImpl, "pcp": AscendMlaPCPImpl, "dcp": AscendMlaDCPImpl}[expected]
    assert AscendMLABackend.get_impl_cls() is expected_cls


@pytest.mark.parametrize("enable_mlapo", [False, True])
def test_inherited_forward_dispatches_to_pcp_and_preserves_standalone_gate(pcp_runtime, monkeypatch, enable_mlapo):
    impl = _impl(pcp_runtime)
    monkeypatch.setattr(mla_v1, "_EXTRA_CTX", SimpleNamespace(num_tokens=2))
    save_kv = Mock()
    monkeypatch.setattr(mla_v1, "maybe_save_kv_layer_to_connector", save_kv)
    impl.enable_mlapo = enable_mlapo
    impl.use_output_gate = True
    impl.g_proj = Mock(return_value=(torch.full((2, NUM_HEADS * VALUE_DIM), 2.0),))
    impl.mla_preprocess_prolog_v3 = Mock(side_effect=AssertionError("PCP must use its own exchange"))
    impl.mla_preprocess_only_decode = Mock(side_effect=AssertionError("PCP decode KV must be gathered"))
    decode = object()
    impl.mla_preprocess_pcp_prolog_v3 = Mock(return_value=(decode, None))
    impl._forward_decode = Mock(return_value=torch.ones(2, NUM_HEADS * VALUE_DIM))
    impl.o_proj = Mock(side_effect=lambda tensor, **kwargs: (tensor,))
    metadata = _metadata(torch.tensor([[0, 1], [2, 3]]), 0, 2, 2)
    hidden = torch.ones(2, 1)
    cache = _cache()
    output = torch.empty(2, NUM_HEADS * VALUE_DIM)
    result = impl.forward(impl.layer_name, hidden, cache, metadata, output)
    assert result is output
    torch.testing.assert_close(output, torch.full_like(output, torch.sigmoid(torch.tensor(2.0))))
    impl.mla_preprocess_pcp_prolog_v3.assert_called_once_with(impl.layer_name, hidden, cache, metadata)
    impl.native_preprocess.assert_not_called()
    impl.mla_preprocess_prolog_v3.assert_not_called()
    impl.mla_preprocess_only_decode.assert_not_called()
    save_kv.assert_called_once_with(impl.layer_name, list(cache))
