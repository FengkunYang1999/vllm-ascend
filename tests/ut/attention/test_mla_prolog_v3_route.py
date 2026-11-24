# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU routing tests for MLA PrologV3 and history cache layouts."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch_npu
from torch import nn
from vllm.model_executor.layers.attention import MLAAttention
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.offloader.base import NoopOffloader
from vllm.model_executor.offloader.prefetch import PrefetchOffloader
from vllm.v1.kv_cache_interface import SlidingWindowMLASpec

import vllm_ascend.attention.mla_v1 as mla_v1
import vllm_ascend.ops.rotary_embedding as rotary_embedding
from vllm_ascend.attention.context_parallel.mla_cp import AscendMlaDCPImpl
from vllm_ascend.attention.mla_v1 import AscendMLAImpl
from vllm_ascend.attention.utils import supports_strided_mla_cache
from vllm_ascend.model_executor.offloader.prefetch import AscendPrefetchOffloader
from vllm_ascend.ops.linear import AscendDCPGroupColumnParallelLinear
from vllm_ascend.quantization.methods import (
    AscendW8A8DynamicLinearMethod,
    AscendW8A8FP8DynamicLinearMethod,
    AscendW8A8LinearMethod,
    AscendW8A8MXFP8DynamicLinearMethod,
)
from vllm_ascend.utils import enable_sfa

NOPE_DIM = 512
ROPE_DIM = 64
FUSED_DIM = NOPE_DIM + ROPE_DIM
BLOCK_SIZE = 128


def _make_impl(impl_type=AscendMLAImpl) -> AscendMLAImpl:
    impl = impl_type.__new__(impl_type)
    impl.layer_name = "model.layers.0.self_attn.attn"
    impl.use_flashmla = False
    impl.use_pcp = False
    impl.use_dcp = impl_type is AscendMlaDCPImpl
    impl.is_pcp_decode_sharded = False
    impl._pcp_prolog_v3_buffers = None
    impl._pcp_prolog_uses_split_history = False
    impl.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=3),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[]),
    )
    impl.dcp_q_replicate_enabled = False
    impl.use_mla_rope = True
    impl.qk_rope_head_dim = ROPE_DIM
    impl.dtype = torch.bfloat16
    impl.enable_kv_nz = False
    impl.fa_quant_layer = False
    impl.support_fp8_attention = False
    impl.num_kv_heads = 1
    impl.num_heads = impl.q_projection_heads = 8
    impl.mlapo_num_heads = 8
    impl.kv_lora_rank = NOPE_DIM
    impl.qk_nope_head_dim = 128
    impl.qk_head_dim = impl.qk_nope_head_dim + ROPE_DIM
    impl.v_head_dim = 128
    impl.q_lora_rank = 128
    impl.fused_qkv_a_proj = SimpleNamespace(weight=MagicMock(), input_size=7168)
    impl.q_proj = SimpleNamespace(weight=MagicMock(), _chunk_size=0)
    impl.q_a_layernorm = SimpleNamespace(
        weight=SimpleNamespace(data=torch.ones(128)),
        variance_epsilon=1e-6,
    )
    impl.kv_a_layernorm = SimpleNamespace(
        weight=SimpleNamespace(data=torch.ones(NOPE_DIM)),
        variance_epsilon=1e-6,
    )
    impl.mlapo_W_UK_T = torch.empty(8, 128, NOPE_DIM)
    impl.kv_b_proj = MagicMock()
    impl.layerwise_kv_cache_hook = None
    impl._prolog_v3_weights = {
        "quant_type": None,
        "weight_dq": torch.empty(128, 128),
        "weight_uq_qr": torch.empty(128, 8 * 192),
        "weight_dkv_kr": torch.empty(128, FUSED_DIM),
        "weight_uk": torch.empty(8, 128, NOPE_DIM),
        "num_heads": 8,
    }
    return impl


def _construct_impl(pcp_size: int, dcp_size: int) -> AscendMLAImpl:
    """Construct the production MLA implementation without weight processing."""
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            runner_type="generate", dtype=torch.float16, hf_config=SimpleNamespace(model_type="deepseek_v3")
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            decode_context_parallel_size=dcp_size,
            pcp_shard_decode_requests=pcp_size > 1 and dcp_size == 1,
        ),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        speculative_config=None,
    )
    with (
        patch.object(mla_v1, "get_current_vllm_config", return_value=config),
        patch.object(
            mla_v1,
            "get_current_hardware_profile",
            return_value=SimpleNamespace(supports=lambda _capability: False),
        ),
        patch.object(mla_v1, "get_ascend_config", return_value=SimpleNamespace(enable_kv_nz=False)),
        patch.object(mla_v1, "enabling_mlapo", return_value=False),
        patch.object(mla_v1, "enable_fa_quant", return_value=False),
    ):
        return AscendMLAImpl(
            num_heads=8,
            head_size=FUSED_DIM,
            scale=1.0,
            num_kv_heads=1,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="auto",
            logits_soft_cap=None,
            attn_type="decoded_attention",
            kv_sharing_target_layer_name=None,
            q_lora_rank=128,
            kv_lora_rank=NOPE_DIM,
            qk_nope_head_dim=128,
            qk_rope_head_dim=ROPE_DIM,
            qk_head_dim=128 + ROPE_DIM,
            v_head_dim=128,
            rotary_emb=None,
            fused_qkv_a_proj=MagicMock(),
            q_b_proj=MagicMock(),
            kv_b_proj=MagicMock(),
            o_proj=None,
            use_mla_rope=True,
        )


def _make_post_load_impl(monkeypatch, impl_type=AscendMLAImpl):
    """Run the actual weight hook with small, already loaded projections."""
    impl = impl_type.__new__(impl_type)
    impl.__dict__.update(_make_impl().__dict__)
    impl.num_heads = impl.num_heads_padded = 2
    impl.head_padding = 0
    impl.mlapo_num_heads = 2
    impl.q_lora_rank = impl.kv_lora_rank = 4
    impl.qk_nope_head_dim = impl.qk_rope_head_dim = 2
    impl.qk_head_dim = 4
    impl.v_head_dim = 2
    impl.enable_mlapo = False
    impl.use_pcp = False
    impl.use_dcp = impl_type is AscendMlaDCPImpl
    impl.is_pcp_decode_sharded = False
    impl.speculative_config = None
    impl._prolog_v3_weights = None
    impl._dcp_current_kv_buffers = None
    impl._dcp_prefill_current_kv_buffers = None
    impl.q_a_layernorm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6)
    impl.kv_a_layernorm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6)
    quant_method = object.__new__(AscendW8A8LinearMethod)
    impl.fused_qkv_a_proj = SimpleNamespace(
        weight=torch.ones(4, 10),
        input_size=7168,
        weight_scale=torch.ones(10),
        quant_method=SimpleNamespace(quant_method=quant_method),
    )
    impl.q_proj = SimpleNamespace(
        weight=torch.ones(4, 8),
        weight_scale=torch.ones(8),
        quant_method=SimpleNamespace(quant_method=quant_method),
        _chunk_size=0,
    )
    impl.kv_b_proj = SimpleNamespace(
        weight=torch.ones(8, 4),
        quant_method=UnquantizedLinearMethod(),
    )
    impl.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16, hf_config=SimpleNamespace(), hf_text_config=SimpleNamespace()
        ),
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE, cache_dtype="auto"),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16, max_num_seqs=4),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=2),
        speculative_config=None,
        kv_transfer_config=None,
        compilation_config=SimpleNamespace(static_forward_context={}, cudagraph_capture_sizes=[]),
        use_v2_model_runner=False,
    )
    owner = MLAAttention.__new__(MLAAttention)
    nn.Module.__init__(owner)
    owner.attn_backend = mla_v1.AscendMLABackend
    owner.impl = impl
    owner.indexer = None
    owner.kv_cache_dtype = "auto"
    owner.head_size = impl.kv_lora_rank + impl.qk_rope_head_dim
    # vLLM main's ``MLAAttention.get_kv_cache_spec`` reads this instance
    # attribute set by ``__init__``; this stub bypasses ``__init__``.
    owner.topk_buffer_width = None
    owner.sliding_window = None
    owner.non_causal_multi_token_decode = False
    impl.vllm_config.compilation_config.static_forward_context[impl.layer_name] = owner
    monkeypatch.setattr(mla_v1, "maybe_trans_nz", lambda tensor: tensor)
    monkeypatch.setattr(torch_npu, "npu_format_cast", lambda tensor, _format: tensor)
    monkeypatch.setattr(mla_v1, "get_offloader", lambda: NoopOffloader())
    monkeypatch.setattr(mla_v1, "is_dcp_enabled", lambda: impl.use_dcp)
    monkeypatch.setattr(
        mla_v1,
        "get_ascend_config",
        lambda: SimpleNamespace(
            sparse_kv_offload_config=SimpleNamespace(enabled=False),
            mlapo_keep_prefill_weights=True,
        ),
    )
    monkeypatch.setattr(
        mla_v1,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(supports=lambda _capability: False),
    )
    return impl


@pytest.mark.parametrize("use_v2_runner", [False, True])
@pytest.mark.parametrize("native_weights", [False, True])
def test_post_load_prepares_private_prolog_weights(monkeypatch, use_v2_runner, native_weights):
    impl = _make_post_load_impl(monkeypatch)
    impl.vllm_config.use_v2_model_runner = use_v2_runner
    if native_weights:
        for projection in (impl.fused_qkv_a_proj, impl.q_proj):
            projection.weight = projection.weight.T.contiguous()
            projection.quant_method = UnquantizedLinearMethod()

    impl.process_weights_after_loading(torch.bfloat16)

    prepared = impl._prolog_v3_weights
    assert prepared is not None
    assert prepared["weight_dq"].shape == (4, 4)
    assert prepared["weight_dkv_kr"].shape == (4, 6)
    assert prepared["weight_uq_qr"].shape == (4, 8)
    assert prepared["weight_uk"] is impl.mlapo_W_UK_T


@pytest.mark.parametrize("use_v2_runner", [False, True])
def test_post_load_prepares_sliding_window_prolog_weights(monkeypatch, use_v2_runner):
    impl = _make_post_load_impl(monkeypatch)
    config = impl.vllm_config
    config.use_v2_model_runner = use_v2_runner
    owner = config.compilation_config.static_forward_context[impl.layer_name]
    owner.sliding_window = BLOCK_SIZE
    assert isinstance(owner.get_kv_cache_spec(config), SlidingWindowMLASpec)

    impl.process_weights_after_loading(torch.bfloat16)

    assert impl._prolog_v3_weights is not None


@pytest.mark.parametrize("free_source_weights,use_pcp", [(False, False), (True, False), (True, True)])
def test_post_load_reuses_mlapo_projection_storage(monkeypatch, free_source_weights, use_pcp):
    impl = _make_post_load_impl(monkeypatch)
    impl.enable_mlapo = True
    if use_pcp:
        impl.use_pcp = True
        impl.vllm_config.parallel_config.prefill_context_parallel_size = 2
        impl.vllm_config.use_v2_model_runner = True
        for projection in (impl.fused_qkv_a_proj, impl.q_proj):
            projection.quant_method.quant_method = object.__new__(AscendW8A8DynamicLinearMethod)
    if free_source_weights:
        impl.vllm_config.kv_transfer_config = SimpleNamespace(kv_connector="MooncakeConnectorV2", is_kv_consumer=True)
        monkeypatch.setattr(
            mla_v1,
            "get_ascend_config",
            lambda: SimpleNamespace(
                sparse_kv_offload_config=SimpleNamespace(enabled=False), mlapo_keep_prefill_weights=False
            ),
        )

    impl.process_weights_after_loading(torch.bfloat16)

    prepared = impl._prolog_v3_weights
    assert prepared is not None
    for name in ("weight_dq", "weight_dkv_kr", "weight_uq_qr", "dequant_scale_w_dq"):
        assert prepared[name] is getattr(impl, name)
    assert prepared["weight_uk"] is impl.mlapo_W_UK_T
    if free_source_weights:
        assert impl.fused_qkv_a_proj.weight is None
        assert impl.q_proj.weight is None
    if use_pcp:
        # MLAPO may already have released source projections on a decode-only consumer.
        nope, rope, indices = impl._pcp_prolog_v3_buffers
        assert nope.device == rope.device == indices.device == impl.W_UV.device
        assert indices.numel() == impl.vllm_config.scheduler_config.max_num_batched_tokens


def test_post_load_refreshes_private_prolog_weights_in_place(monkeypatch):
    impl = _make_post_load_impl(monkeypatch)
    impl.process_weights_after_loading(torch.bfloat16)
    prepared = impl._prolog_v3_weights
    assert prepared is not None
    pointers = {name: value.data_ptr() for name, value in prepared.items() if isinstance(value, torch.Tensor)}
    impl.fused_qkv_a_proj.weight = torch.full((4, 10), 2.0)
    impl.q_proj.weight = torch.full((4, 8), 3.0)
    impl.fused_qkv_a_proj.weight_scale = torch.full((10,), 4.0)
    impl.q_proj.weight_scale = torch.full((8,), 5.0)

    impl.process_weights_after_loading(torch.bfloat16)

    assert impl._prolog_v3_weights is prepared
    for name, pointer in pointers.items():
        assert prepared[name].data_ptr() == pointer
    torch.testing.assert_close(prepared["weight_dq"], torch.full((4, 4), 2.0))
    torch.testing.assert_close(prepared["weight_dkv_kr"], torch.full((4, 6), 2.0))
    torch.testing.assert_close(prepared["weight_uq_qr"], torch.full((4, 8), 3.0))
    torch.testing.assert_close(prepared["dequant_scale_w_dq"], torch.full((1, 4), 4.0))
    torch.testing.assert_close(prepared["dequant_scale_w_uq_qr"], torch.full((1, 8), 5.0))


@pytest.mark.parametrize("offloader_type", [PrefetchOffloader, AscendPrefetchOffloader])
def test_post_load_offloader_keeps_prolog_weights_lazy(monkeypatch, offloader_type):
    impl = _make_post_load_impl(monkeypatch)
    offloader = offloader_type.__new__(offloader_type)
    monkeypatch.setattr(mla_v1, "get_offloader", lambda: offloader)

    impl.process_weights_after_loading(torch.bfloat16)

    assert impl._prolog_v3_weights is None


@pytest.mark.parametrize(
    "excluded_path",
    [
        "fp16",
        "no_rope",
        "nz",
        "indexer",
        "sparse_offload",
        "compress",
        "fa_quant",
        "unsupported_quant",
        "model_version",
        "block_stride",
        "unsupported_pd",
    ],
)
def test_post_load_skips_unused_prolog_weights(monkeypatch, excluded_path):
    impl = _make_post_load_impl(monkeypatch)
    config = impl.vllm_config
    owner = config.compilation_config.static_forward_context[impl.layer_name]
    if excluded_path == "fp16":
        impl.dtype = torch.float16
    elif excluded_path == "no_rope":
        impl.use_mla_rope = False
    elif excluded_path == "nz":
        impl.enable_kv_nz = True
    elif excluded_path == "indexer":
        owner.indexer = nn.Identity()
    elif excluded_path == "sparse_offload":
        monkeypatch.setattr(
            mla_v1,
            "get_ascend_config",
            lambda: SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=True)),
        )
    elif excluded_path == "compress":
        config.model_config.hf_config.compress_ratios = [1]
    elif excluded_path == "fa_quant":
        impl.fa_quant_layer = True
        impl._mlapo_uses_native_weights = False
        owner.quant_kscale = torch.ones(1)
        owner.fak_descale_float = torch.ones(1)
        owner.fak_descale_reciprocal = torch.ones(1)
    elif excluded_path == "unsupported_quant":
        for projection in (impl.fused_qkv_a_proj, impl.q_proj):
            projection.quant_method.quant_method = object.__new__(AscendW8A8FP8DynamicLinearMethod)
    elif excluded_path == "model_version":
        owner.model_version = "glm5_next"
    elif excluded_path == "block_stride":
        owner.indexes_kv_by_block_stride = True
    else:
        config.kv_transfer_config = SimpleNamespace(kv_connector="UnsupportedConnector")

    impl.process_weights_after_loading(torch.bfloat16)

    assert impl._prolog_v3_weights is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("tokens_per_state", 2),
        ("model_version", "deepseek_v4"),
        ("indexes_kv_by_block_stride", True),
        ("cache_sparse_sfa_c8", True),
    ],
)
def test_post_load_generic_spec_uses_same_layout_exclusions(monkeypatch, field, value):
    impl = _make_post_load_impl(monkeypatch)
    config = impl.vllm_config
    owner = config.compilation_config.static_forward_context[impl.layer_name]
    cache_spec = owner.get_kv_cache_spec(config)
    assert supports_strided_mla_cache(config, owner, cache_spec)
    object.__setattr__(cache_spec, field, value)

    assert not supports_strided_mla_cache(config, owner, cache_spec)


@pytest.mark.parametrize("use_v2_runner", [False, True])
def test_dcp_post_load_prepares_prolog_weights_after_current_kv_buffers(monkeypatch, use_v2_runner):
    impl = _make_post_load_impl(monkeypatch, AscendMlaDCPImpl)
    impl.vllm_config.use_v2_model_runner = use_v2_runner
    prepare = impl._prepare_prolog_v3_weights

    def prepare_after_buffers():
        assert impl._dcp_current_kv_buffers_are_sufficient()
        return prepare()

    impl._prepare_prolog_v3_weights = MagicMock(side_effect=prepare_after_buffers)

    impl.process_weights_after_loading(torch.bfloat16)

    impl._prepare_prolog_v3_weights.assert_called_once()
    assert impl._prolog_v3_weights is not None
    assert impl._dcp_current_kv_buffers is not None
    assert impl._dcp_prefill_current_kv_buffers is not None


@pytest.mark.parametrize(
    ("pcp_size", "dcp_size", "expected_pcp", "expected_dcp"),
    [
        (1, 1, False, False),
        (1, 2, False, True),
        (2, 1, True, False),
        (2, 2, True, True),
    ],
)
def test_constructor_initializes_parallel_routing_flags(
    pcp_size,
    dcp_size,
    expected_pcp,
    expected_dcp,
):
    impl = _construct_impl(pcp_size, dcp_size)

    assert impl.use_pcp is expected_pcp
    assert impl.use_dcp is expected_dcp
    assert impl.is_pcp_decode_sharded is (expected_pcp and dcp_size == 1)


def _component_major_cache(blocks: int = 4, dtype=torch.float32):
    page = BLOCK_SIZE * FUSED_DIM
    raw = torch.arange(blocks * 2 * page, dtype=dtype).reshape(blocks * 2, page)
    nope = raw.as_strided(
        (blocks, BLOCK_SIZE, 1, NOPE_DIM),
        (2 * page, NOPE_DIM, NOPE_DIM, 1),
        0,
    )
    rope = raw.as_strided(
        (blocks, BLOCK_SIZE, 1, ROPE_DIM),
        (2 * page, ROPE_DIM, ROPE_DIM, 1),
        BLOCK_SIZE * NOPE_DIM,
    )
    return nope, rope


def _token_fused_cache(blocks: int = 4):
    fused = torch.arange(blocks * BLOCK_SIZE * FUSED_DIM, dtype=torch.float32).reshape(blocks, BLOCK_SIZE, 1, FUSED_DIM)
    nope = fused[..., :NOPE_DIM]
    rope = fused[..., NOPE_DIM:]
    return nope, rope


def _load_cache_rows_on_cpu(cache_k, cache_v, slots):
    packed = cache_k.new_zeros((slots.numel(), cache_k.shape[-1] + cache_v.shape[-1]))
    for row, slot in enumerate(slots.reshape(-1).tolist()):
        if slot >= 0:
            block, offset = divmod(slot, cache_k.shape[1])
            packed[row, : cache_k.shape[-1]] = cache_k[block, offset, 0]
            packed[row, cache_k.shape[-1] :] = cache_v[block, offset, 0]
    return packed


@pytest.mark.parametrize("cache_factory", [_component_major_cache, _token_fused_cache])
@pytest.mark.parametrize("slots", [[], [0], [129, -1, 0, 129, 3]])
def test_current_kv_gather_materializes_only_selected_rows(cache_factory, slots):
    cache_k, cache_v = cache_factory()
    slot_tensor = torch.tensor(slots, dtype=torch.int64)
    packed = _load_cache_rows_on_cpu(cache_k, cache_v, slot_tensor)

    class CacheReadGuard(torch.Tensor):
        def __getitem__(self, index):
            raise AssertionError("Do not index the entire strided cache.")

        def contiguous(self, *args, **kwargs):
            raise AssertionError("Do not materialize the entire strided cache.")

    guarded_cache = (cache_k.as_subclass(CacheReadGuard), cache_v.as_subclass(CacheReadGuard))
    with patch.object(mla_v1, "load_mla_cache_rows_by_slots", return_value=packed) as mock_reader:
        nope, rope = AscendMLAImpl._gather_current_kv_from_cache(guarded_cache, slot_tensor)

    mock_reader.assert_called_once_with(*guarded_cache, slot_tensor)
    assert nope.shape == (len(slots), 1, NOPE_DIM)
    assert rope.shape == (len(slots), 1, ROPE_DIM)
    assert nope.is_contiguous() and rope.is_contiguous()
    torch.testing.assert_close(nope[:, 0], packed[:, :NOPE_DIM])
    torch.testing.assert_close(rope[:, 0], packed[:, NOPE_DIM:])
    # Patching Tensor.__getitem__ leaves CPython sequence slots changed even
    # after restoration, breaking scalar-tensor construction in later EPLB UTs.
    torch.testing.assert_close(
        torch.tensor([torch.tensor(3), torch.tensor(7)], dtype=torch.int32),
        torch.tensor([3, 7], dtype=torch.int32),
    )


@pytest.mark.parametrize("cache_factory", [_component_major_cache, _token_fused_cache])
def test_strided_mla_caches_require_prolog_v3(cache_factory):
    impl = _make_impl()
    nope, rope = cache_factory()
    assert impl._cache_requires_prolog_v3_writer((nope, rope))


def test_contiguous_mla_cache_does_not_force_prolog_v3():
    impl = _make_impl()
    nope, rope = _component_major_cache()
    assert not impl._cache_requires_prolog_v3_writer((nope.contiguous(), rope.contiguous()))


def test_prolog_v3_weights_pad_native_query_heads(monkeypatch):
    impl = _make_impl()
    impl._prolog_v3_weights = None
    impl.num_heads = 3
    impl.num_heads_padded = 4
    impl.head_padding = 1
    impl.qk_head_dim = impl.qk_nope_head_dim + ROPE_DIM
    impl.mlapo_W_UK_T = torch.arange(3 * impl.qk_nope_head_dim * NOPE_DIM, dtype=torch.float32).reshape(
        3, impl.qk_nope_head_dim, NOPE_DIM
    )
    impl.fused_qkv_a_proj = SimpleNamespace(
        weight=SimpleNamespace(data=torch.randn(128 + FUSED_DIM, 128)),
        quant_method=SimpleNamespace(quant_method=None),
    )
    impl.q_proj = SimpleNamespace(
        weight=SimpleNamespace(data=torch.randn(3 * impl.qk_head_dim, 128)),
        _chunk_size=0,
    )
    monkeypatch.setattr("vllm_ascend.attention.mla_v1.torch_npu.npu_format_cast", lambda tensor, _: tensor)

    prepared = impl._prepare_prolog_v3_weights()

    assert prepared["num_heads"] == 4
    assert prepared["weight_uq_qr"].shape == (128, 4 * impl.qk_head_dim)
    assert prepared["weight_uk"].shape == (4, impl.qk_nope_head_dim, NOPE_DIM)


def test_prolog_v3_weights_reuse_prepared_weights_after_source_free():
    impl = _make_impl()
    impl._prolog_v3_weights = None
    quant_method = object.__new__(AscendW8A8LinearMethod)
    impl._get_context_prolog_quant_method = lambda _layer: quant_method

    weight_dq = torch.empty(1)
    weight_dkv_kr = torch.empty(2)
    weight_uq_qr = torch.empty(3)
    impl.weight_dq = weight_dq
    impl.weight_dkv_kr = weight_dkv_kr
    impl.weight_uq_qr = weight_uq_qr
    impl.dequant_scale_w_dq = torch.empty(4)
    impl.dequant_scale_w_dkv_kr = torch.empty(5)
    impl.dequant_scale_w_uq_qr = torch.empty(6)
    impl.fused_qkv_a_proj.weight = None
    impl.q_proj.weight = None

    prepared = impl._prepare_prolog_v3_weights()

    assert prepared["weight_dq"] is weight_dq
    assert prepared["weight_dkv_kr"] is weight_dkv_kr
    assert prepared["weight_uq_qr"] is weight_uq_qr
    assert prepared["dequant_scale_w_dq"] is impl.dequant_scale_w_dq


def test_prolog_v3_weights_accept_static_w8a8(monkeypatch):
    impl = _make_impl()
    impl._prolog_v3_weights = None
    quant_method = object.__new__(AscendW8A8LinearMethod)
    impl._get_context_prolog_quant_method = lambda _layer: quant_method
    fused_output_dim = impl.q_lora_rank + FUSED_DIM
    q_output_dim = impl.num_heads * (impl.qk_nope_head_dim + ROPE_DIM)
    impl.fused_qkv_a_proj = SimpleNamespace(
        weight=SimpleNamespace(data=torch.zeros(impl.q_lora_rank, fused_output_dim)),
        weight_scale=torch.ones(fused_output_dim),
        quant_method=SimpleNamespace(quant_method=quant_method),
    )
    impl.q_proj = SimpleNamespace(
        weight=SimpleNamespace(data=torch.zeros(impl.q_lora_rank, q_output_dim)),
        weight_scale=torch.ones(q_output_dim),
        _chunk_size=0,
    )
    monkeypatch.setattr("vllm_ascend.attention.mla_v1.torch_npu.npu_format_cast", lambda tensor, _: tensor)

    prepared = impl._prepare_prolog_v3_weights()

    assert prepared["quant_type"] is AscendW8A8LinearMethod
    assert prepared["weight_dq"].shape == (impl.q_lora_rank, impl.q_lora_rank)
    assert prepared["weight_dkv_kr"].shape == (impl.q_lora_rank, FUSED_DIM)
    assert prepared["weight_uq_qr"].shape == (impl.q_lora_rank, q_output_dim)
    assert prepared["dequant_scale_w_dq"].shape == (1, impl.q_lora_rank)
    assert prepared["dequant_scale_w_dkv_kr"].shape == (1, FUSED_DIM)
    assert prepared["dequant_scale_w_uq_qr"].shape == (1, q_output_dim)


@pytest.mark.parametrize("use_pcp", [False, True])
@pytest.mark.parametrize("quant_type", [AscendW8A8LinearMethod, AscendW8A8DynamicLinearMethod])
def test_pcp_quantization_admission_preserves_non_pcp_writers(use_pcp, quant_type):
    impl = _make_impl()
    impl.use_pcp = use_pcp
    for projection in (impl.fused_qkv_a_proj, impl.q_proj):
        projection.quant_method = SimpleNamespace(quant_method=object.__new__(quant_type))

    supported = not use_pcp or quant_type is AscendW8A8DynamicLinearMethod
    assert impl.supports_prolog_v3_writer() is supported
    assert impl._cache_requires_prolog_v3_writer(_component_major_cache()) is supported


def test_prolog_v3_weights_keep_int8_dynamic_scale_layout_on_a5(monkeypatch):
    impl = _make_impl()
    impl._prolog_v3_weights = None
    impl.support_fp8_attention = True
    quant_method = object.__new__(AscendW8A8DynamicLinearMethod)
    impl._get_context_prolog_quant_method = lambda _layer: quant_method
    fused_output_dim = impl.q_lora_rank + FUSED_DIM
    q_output_dim = impl.num_heads * (impl.qk_nope_head_dim + ROPE_DIM)
    impl.fused_qkv_a_proj = SimpleNamespace(
        weight=SimpleNamespace(data=torch.zeros(impl.q_lora_rank, fused_output_dim)),
        weight_scale=torch.ones(fused_output_dim),
        quant_method=SimpleNamespace(quant_method=quant_method),
    )
    impl.q_proj = SimpleNamespace(
        weight=SimpleNamespace(data=torch.zeros(impl.q_lora_rank, q_output_dim)),
        weight_scale=torch.ones(q_output_dim),
        _chunk_size=0,
    )
    monkeypatch.setattr("vllm_ascend.attention.mla_v1.torch_npu.npu_format_cast", lambda tensor, _: tensor)

    prepared = impl._prepare_prolog_v3_weights()

    assert prepared["dequant_scale_w_dq"].shape == (1, impl.q_lora_rank)
    assert prepared["dequant_scale_w_dkv_kr"].shape == (1, FUSED_DIM)
    assert prepared["dequant_scale_w_uq_qr"].shape == (1, q_output_dim)


@pytest.mark.parametrize("use_pcp", [False, True])
@pytest.mark.parametrize(
    "internal_sources", [(False, False), (True, False), (False, True), (True, True)], ids=["dense", "a", "qb", "both"]
)
@pytest.mark.parametrize(
    "quant_type,dtype",
    [
        (None, torch.bfloat16),
        (AscendW8A8LinearMethod, torch.int8),
        (AscendW8A8DynamicLinearMethod, torch.int8),
        (AscendW8A8MXFP8DynamicLinearMethod, torch.float8_e4m3fn),
    ],
    ids=["native", "static-int8", "dynamic-int8", "mxfp8"],
)
def test_prolog_weights_normalize_internal_sources_before_materialization(
    monkeypatch, use_pcp, internal_sources, quant_type, dtype
):
    """Exercise shared preparation, independently of each scheme's writer admission."""
    impl = _make_post_load_impl(monkeypatch)
    impl.use_pcp = use_pcp
    impl.mlapo_W_UK_T = torch.zeros(impl.num_heads, impl.qk_nope_head_dim, impl.kv_lora_rank)
    quant_method = None if quant_type is None else object.__new__(quant_type)
    impl._get_context_prolog_quant_method = lambda _layer: quant_method

    class InternalWeight(torch.Tensor):
        def contiguous(self, *args, **kwargs):
            raise RuntimeError("Internal-format weights must be converted to ND before materialization.")

    source_layouts = {}

    def format_cast(tensor, target_format):
        if target_format == mla_v1.ACL_FORMAT_FRACTAL_ND:
            # ND sources must not be converted. Internal sources must reach
            # this boundary whole, before slicing or the native transpose.
            assert isinstance(tensor, InternalWeight)
            assert (tensor.shape, tensor.stride()) == source_layouts[tensor.data_ptr()]
            return tensor.as_subclass(torch.Tensor).clone()
        assert target_format == mla_v1.ACL_FORMAT_FRACTAL_NZ
        assert not isinstance(tensor, InternalWeight)
        return tensor.clone()

    monkeypatch.setattr(mla_v1, "is_internal_npu_format", lambda tensor: isinstance(tensor, InternalWeight))
    monkeypatch.setattr(torch_npu, "npu_format_cast", format_cast)
    fused_dim = impl.q_lora_rank + impl.kv_lora_rank + impl.qk_rope_head_dim
    q_dim = impl.num_heads * impl.qk_head_dim
    fused_values = torch.arange(impl.q_lora_rank * fused_dim).reshape(impl.q_lora_rank, fused_dim) % 15 - 7
    q_values = torch.arange(impl.q_lora_rank * q_dim).reshape(impl.q_lora_rank, q_dim) % 15 - 7
    # Rebuild one internal INT8 case as well: reload must repeat normalization
    # while retaining every graph-visible buffer allocated by the first build.
    reload_weights = use_pcp and all(internal_sources) and quant_type is AscendW8A8DynamicLinearMethod
    for generation in range(2 if reload_weights else 1):
        logical_weights = [(values + generation).to(dtype) for values in (fused_values, q_values)]
        sources = []
        for projection, values, internal in zip(
            (impl.fused_qkv_a_proj, impl.q_proj), logical_weights, internal_sources
        ):
            storage = values.T.contiguous() if quant_type is None else values
            source = nn.Parameter(storage.as_subclass(InternalWeight) if internal else storage, requires_grad=False)
            projection.weight = source
            source_layouts[source.data_ptr()] = (source.shape, source.stride())
            sources.append((projection, source, storage.clone(), source.data_ptr()))
            output_dim = values.shape[1]
            projection.weight_scale = (
                torch.arange(output_dim * 2, dtype=torch.float32).reshape(1, output_dim, 2) + 1
                if quant_type is AscendW8A8MXFP8DynamicLinearMethod
                else torch.arange(output_dim, dtype=torch.float32) + 1
            )

        if generation == 0:
            prepared = impl._prepare_prolog_v3_weights()
            pointers = {name: value.data_ptr() for name, value in prepared.items() if isinstance(value, torch.Tensor)}
        else:
            impl._refresh_prolog_v3_weights()
            assert impl._prolog_v3_weights is prepared
            assert {name: prepared[name].data_ptr() for name in pointers} == pointers

        expected_weights = {
            "weight_dq": logical_weights[0][:, : impl.q_lora_rank],
            "weight_dkv_kr": logical_weights[0][:, impl.q_lora_rank :],
            "weight_uq_qr": logical_weights[1],
        }
        for name, expected in expected_weights.items():
            assert prepared[name].dtype == dtype
            torch.testing.assert_close(prepared[name].float(), expected.float(), rtol=0, atol=0)
        for projection, source, original, pointer in sources:
            assert projection.weight is source and source.data_ptr() == pointer
            torch.testing.assert_close(source.as_subclass(torch.Tensor).float(), original.float(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "fused_method,q_method",
    [
        (object.__new__(AscendW8A8FP8DynamicLinearMethod), object.__new__(AscendW8A8FP8DynamicLinearMethod)),
        (object.__new__(AscendW8A8DynamicLinearMethod), None),
        (None, object.__new__(AscendW8A8DynamicLinearMethod)),
    ],
)
def test_unsupported_or_mixed_prolog_quantization_keeps_legacy_path(fused_method, q_method):
    impl = _make_impl()
    impl.fused_qkv_a_proj = SimpleNamespace(quant_method=SimpleNamespace(quant_method=fused_method))
    impl.q_proj = SimpleNamespace(quant_method=SimpleNamespace(quant_method=q_method), _chunk_size=0)
    nope, rope = _component_major_cache()

    assert not impl.supports_prolog_v3_quantization()
    assert not impl._cache_requires_prolog_v3_writer((nope, rope))


def test_mla_without_q_lora_keeps_legacy_writer():
    impl = _make_impl()
    impl.q_lora_rank = None
    impl.fused_qkv_a_proj = None
    impl.q_a_layernorm = None
    nope, rope = _component_major_cache()

    assert not impl.supports_prolog_v3_projection()
    assert not impl._cache_requires_prolog_v3_writer((nope, rope))


def test_prolog_v3_writer_requires_rope_and_operator(monkeypatch):
    impl = _make_impl()
    nope, rope = _component_major_cache()

    assert impl.supports_prolog_v3_writer()
    assert impl._cache_requires_prolog_v3_writer((nope, rope))

    impl.use_mla_rope = False
    assert not impl.supports_prolog_v3_writer()
    assert not impl._cache_requires_prolog_v3_writer((nope, rope))
    impl.use_mla_rope = True

    monkeypatch.delattr(torch_npu, "npu_mla_prolog_v3")
    assert not impl.supports_prolog_v3_writer()
    assert not impl._cache_requires_prolog_v3_writer((nope, rope))


def test_prolog_v3_writer_excludes_native_fp16():
    impl = _make_impl()
    nope, rope = _component_major_cache()
    impl.dtype = torch.float16

    assert not impl.supports_prolog_v3_dtype()
    assert not impl.supports_prolog_v3_writer()
    assert not impl._cache_requires_prolog_v3_writer((nope, rope))


def test_dcp_current_kv_capacity_uses_decode_query_len():
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=32, max_num_batched_tokens=16384),
    )
    impl.speculative_config = SimpleNamespace(num_speculative_tokens=3)

    assert impl._required_dcp_current_kv_capacity() == 128

    impl.vllm_config.scheduler_config.max_num_batched_tokens = 96
    assert impl._required_dcp_current_kv_capacity() == 96


def test_dcp_prefill_current_kv_capacity_uses_batched_tokens():
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=32, max_num_batched_tokens=4096),
    )

    assert impl._required_dcp_prefill_current_kv_capacity() == 4096


@pytest.mark.parametrize(
    ("use_dcp", "use_pcp", "decode_sharded", "use_rope", "expected_buffers"),
    [
        (False, False, False, True, "none"),
        (True, False, False, True, "both"),
        (True, True, False, True, "both"),
        (True, False, True, True, "both"),
        (False, False, False, False, "none"),
        (True, False, False, False, "decode"),
        (True, True, False, False, "decode"),
    ],
)
def test_dcp_post_load_allocates_only_consumed_workspaces(use_dcp, use_pcp, decode_sharded, use_rope, expected_buffers):
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192, max_num_seqs=32),
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
    )
    impl.speculative_config = SimpleNamespace(num_speculative_tokens=7)
    impl.use_dcp = use_dcp
    impl.use_pcp = use_pcp
    impl.is_pcp_decode_sharded = decode_sharded
    impl.enable_mlapo = True
    impl.fa_quant_layer = False
    impl.use_mla_rope = use_rope
    impl.enable_kv_nz = False
    impl.num_kv_heads = 1
    impl.kv_lora_rank = NOPE_DIM
    impl.qk_rope_head_dim = ROPE_DIM
    impl.dtype = torch.bfloat16
    impl.fused_qkv_a_proj = SimpleNamespace(weight=torch.empty(1))
    impl._dcp_current_kv_buffers = None
    impl._dcp_prefill_current_kv_buffers = None
    impl.supports_prolog_v3_writer = lambda: use_rope

    with (
        patch.object(AscendMLAImpl, "process_weights_after_loading"),
        patch.object(AscendMLAImpl, "_prepare_prolog_v3_weights_after_loading"),
    ):
        impl.process_weights_after_loading(torch.bfloat16)

    assert (impl._dcp_current_kv_buffers is not None) is (expected_buffers != "none")
    assert (impl._dcp_prefill_current_kv_buffers is not None) is (expected_buffers == "both")
    if expected_buffers != "none":
        assert impl._dcp_current_kv_buffers is not None
        assert impl._dcp_current_kv_buffers[2].numel() == 256
    if expected_buffers == "both":
        assert impl._dcp_prefill_current_kv_buffers is not None
        assert impl._dcp_prefill_current_kv_buffers[2].numel() == 8192


@pytest.mark.parametrize("block_size", [128, 1536])
def test_dcp_current_kv_buffers_survive_reloads(block_size):
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=256, max_num_seqs=256),
        cache_config=SimpleNamespace(block_size=block_size),
    )
    impl.speculative_config = None
    impl.num_kv_heads = 1
    impl.kv_lora_rank = NOPE_DIM
    impl.qk_rope_head_dim = ROPE_DIM
    impl.dtype = torch.float32
    impl.fused_qkv_a_proj = SimpleNamespace(weight=torch.empty(1))
    impl._dcp_current_kv_buffers = None
    impl._dcp_prefill_current_kv_buffers = None
    impl.supports_prolog_v3_writer = lambda: False

    impl._prepare_dcp_current_kv_buffers()
    first_buffers = impl._dcp_current_kv_buffers
    assert first_buffers is not None
    assert first_buffers[0].shape == (2, 128, 1, NOPE_DIM)
    assert first_buffers[1].shape == (2, 128, 1, ROPE_DIM)
    assert first_buffers[2].tolist() == list(range(256))

    impl._prepare_dcp_current_kv_buffers()
    assert impl._dcp_current_kv_buffers is first_buffers
    assert all(
        first.data_ptr() == second.data_ptr()
        for first, second in zip(first_buffers, impl._dcp_current_kv_buffers, strict=True)
    )


@pytest.mark.parametrize("block_size", [128, 1536])
def test_dcp_prolog_buffers_survive_reloads_and_use_disjoint_workspaces(block_size):
    impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    impl.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096, max_num_seqs=32),
        cache_config=SimpleNamespace(block_size=block_size),
    )
    impl.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    impl.num_kv_heads = 1
    impl.kv_lora_rank = NOPE_DIM
    impl.qk_rope_head_dim = ROPE_DIM
    impl.dtype = torch.float32
    impl.fused_qkv_a_proj = SimpleNamespace(weight=torch.empty(1))
    impl._dcp_current_kv_buffers = None
    impl._dcp_prefill_current_kv_buffers = None
    impl.supports_prolog_v3_writer = lambda: True

    impl._prepare_dcp_current_kv_buffers()
    first_decode_buffers = impl._dcp_current_kv_buffers
    first_prefill_buffers = impl._dcp_prefill_current_kv_buffers
    assert first_decode_buffers is not None
    assert first_prefill_buffers is not None
    assert first_decode_buffers[0].shape == (1, 128, 1, NOPE_DIM)
    assert first_prefill_buffers[0].shape == (32, 128, 1, NOPE_DIM)
    assert first_decode_buffers[0].untyped_storage() is not first_prefill_buffers[0].untyped_storage()
    assert impl._dcp_current_kv_buffers_are_sufficient()

    impl._prepare_dcp_current_kv_buffers()
    assert impl._dcp_current_kv_buffers is first_decode_buffers
    assert impl._dcp_prefill_current_kv_buffers is first_prefill_buffers


@pytest.mark.parametrize(
    "num_decode,padding,layerwise",
    [(0, 0, False), (0, 3, False), (2, 0, False), (2, 3, False), (2, 0, True)],
)
def test_mla_preprocess_prolog_v3_replaces_rmsnorm_rope_cache_writer(num_decode, padding, layerwise):
    impl = _make_impl()
    order = []
    if layerwise:
        impl.layerwise_kv_cache_hook = SimpleNamespace(
            layerwise=True, wait_for_layer=lambda name: order.append(("broadcast", name))
        )
    num_prefill = 3
    total_tokens = num_decode + num_prefill + padding
    qkv_lora = torch.zeros(total_tokens, impl.q_lora_rank + FUSED_DIM)
    impl.fused_qkv_a_proj = MagicMock(return_value=(qkv_lora,))
    impl.q_a_layernorm = MagicMock(side_effect=lambda tensor: tensor)
    impl.q_proj = MagicMock(
        side_effect=lambda tensor: (torch.zeros(tensor.shape[0], impl.num_heads * (impl.qk_nope_head_dim + ROPE_DIM)),)
    )
    impl.rope_single = MagicMock(side_effect=lambda tensor, _cos, _sin: tensor)
    nope, rope = _component_major_cache()
    nope.fill_(1.0)
    rope.fill_(2.0)

    hidden = torch.randn(total_tokens, 128)
    decode_cos = torch.randn(num_decode, ROPE_DIM)
    decode_sin = torch.randn(num_decode, ROPE_DIM)
    prefill_cos = torch.randn(num_prefill + padding, 1, 1, ROPE_DIM)
    prefill_sin = torch.randn(num_prefill + padding, 1, 1, ROPE_DIM)
    original_cos, original_sin = prefill_cos.clone(), prefill_sin.clone()
    metadata = SimpleNamespace(
        external_flashmla=None,
        num_decode_tokens=num_decode,
        num_actual_tokens=num_decode + num_prefill,
        slot_mapping=torch.tensor([0, 1][:num_decode] + [129, 130, 131] + [-1] * padding),
        decode=SimpleNamespace(cos=decode_cos, sin=decode_sin),
        prefill=SimpleNamespace(cos=prefill_cos, sin=prefill_sin),
    )

    def fake_prolog(**kwargs):
        if layerwise:
            order.append(("prolog", impl.layer_name))
        tokens = kwargs["token_x"].shape[0]
        q_rows = kwargs["token_x"][:, :1]
        return (
            q_rows.expand(tokens, impl.num_heads * NOPE_DIM),
            (-q_rows).expand(tokens, impl.num_heads * ROPE_DIM),
            None,
            None,
            None,
        )

    projected = torch.zeros(num_prefill, impl.num_heads * (impl.qk_nope_head_dim + impl.v_head_dim))
    impl.kv_b_proj.return_value = (projected,)
    impl._prepare_prolog_v3_weights = lambda: impl._prolog_v3_weights

    with (
        patch("torch_npu.npu_mla_prolog_v3", side_effect=fake_prolog) as mock_prolog,
        patch("torch_npu.npu_kv_rmsnorm_rope_cache") as mock_legacy,
        patch.object(mla_v1, "load_mla_cache_rows_by_slots", side_effect=_load_cache_rows_on_cpu),
        patch.object(
            mla_v1, "wait_for_kv_layer_from_connector", side_effect=lambda name: order.append(("load", name))
        ) as wait_load,
    ):
        decode, prefill = impl.mla_preprocess_prolog_v3(hidden, (nope, rope), metadata)

    wait_load.assert_called_once_with(impl.layer_name)
    if layerwise:
        assert order == [("load", impl.layer_name), ("broadcast", impl.layer_name)] + [("prolog", impl.layer_name)] * (
            1 + bool(num_decode)
        )
    assert mock_prolog.call_count == 1 + bool(num_decode)
    mock_legacy.assert_not_called()
    assert (decode is not None) == bool(num_decode)
    if num_decode:
        decode_kwargs = mock_prolog.call_args_list[0].kwargs
        torch.testing.assert_close(decode_kwargs["token_x"], hidden[:num_decode])
        torch.testing.assert_close(decode_kwargs["cache_index"], metadata.slot_mapping[:num_decode])
        torch.testing.assert_close(decode_kwargs["rope_cos"], decode_cos)
        torch.testing.assert_close(decode_kwargs["rope_sin"], decode_sin)
        torch.testing.assert_close(decode.ql_nope[:, 0, 0], hidden[:num_decode, 0])
        torch.testing.assert_close(decode.q_pe[:, 0, 0], -hidden[:num_decode, 0])
    assert prefill is not None
    assert prefill.q_nope.shape == (num_prefill, impl.num_heads, impl.qk_nope_head_dim)
    assert prefill.q_pe.shape == (num_prefill, impl.num_heads, ROPE_DIM)
    assert prefill.k_pe.shape == (num_prefill, impl.num_heads, ROPE_DIM)
    assert torch.all(prefill.k_pe == 2.0)
    impl.rope_single.assert_called_once()
    rope_input, rope_cos, rope_sin = impl.rope_single.call_args.args
    assert rope_input.shape[0] == num_prefill
    torch.testing.assert_close(rope_cos, original_cos[:num_prefill])
    torch.testing.assert_close(rope_sin, original_sin[:num_prefill])
    prefill_kwargs = mock_prolog.call_args_list[-1].kwargs
    actual = num_decode + num_prefill
    torch.testing.assert_close(prefill_kwargs["token_x"], hidden[num_decode:actual])
    torch.testing.assert_close(prefill_kwargs["cache_index"], metadata.slot_mapping[num_decode:actual])
    torch.testing.assert_close(prefill_kwargs["rope_cos"], original_cos[:num_prefill].reshape(num_prefill, ROPE_DIM))
    torch.testing.assert_close(prefill_kwargs["rope_sin"], original_sin[:num_prefill].reshape(num_prefill, ROPE_DIM))
    assert metadata.prefill.cos is prefill_cos and metadata.prefill.sin is prefill_sin
    torch.testing.assert_close(metadata.prefill.cos, original_cos)
    torch.testing.assert_close(metadata.prefill.sin, original_sin)


def test_non_pcp_prolog_does_not_publish_padding_only_batch():
    impl = _make_impl()
    metadata = SimpleNamespace(
        num_actual_tokens=0,
        num_decode_tokens=0,
        slot_mapping=torch.full((3,), -1),
        external_flashmla=None,
        decode=None,
        prefill=None,
    )
    with (
        patch("torch_npu.npu_mla_prolog_v3") as prolog,
        patch.object(mla_v1, "get_pcp_group") as group,
        patch.object(mla_v1, "notify_kv_cache_written") as notify,
    ):
        assert impl.mla_preprocess_prolog_v3(torch.zeros(3, 128), _component_major_cache(), metadata) == (None, None)
    prolog.assert_not_called()
    group.assert_not_called()
    notify.assert_not_called()


@pytest.mark.parametrize(
    (
        "use_dcp",
        "has_dcp_buffers",
        "use_pcp",
        "is_pcp_decode_sharded",
        "requires_current_kv",
        "expects_prolog",
    ),
    [
        (False, False, False, False, False, True),
        (True, True, False, False, False, True),
        (True, True, False, False, True, True),
        (True, False, False, False, False, False),
        (False, False, False, False, True, False),
        (False, False, True, False, False, True),
        (False, False, False, True, False, True),
    ],
)
def test_forward_routes_strided_cache_with_runtime_exclusions(
    use_dcp,
    has_dcp_buffers,
    use_pcp,
    is_pcp_decode_sharded,
    requires_current_kv,
    expects_prolog,
):
    impl = _make_impl(AscendMlaDCPImpl if use_dcp else AscendMLAImpl)
    impl.use_output_gate = False
    impl.enable_mlapo = False
    impl.use_dcp = use_dcp
    if has_dcp_buffers:
        impl._dcp_current_kv_buffers = (
            torch.empty(1, BLOCK_SIZE, 1, NOPE_DIM),
            torch.empty(1, BLOCK_SIZE, 1, ROPE_DIM),
            torch.arange(BLOCK_SIZE),
        )
    impl.use_pcp = use_pcp
    impl.is_pcp_decode_sharded = is_pcp_decode_sharded
    impl._decode_requires_current_kv = MagicMock(return_value=requires_current_kv)
    impl._prefill_requires_current_kv = MagicMock(return_value=False)
    impl._cache_requires_prolog_v3_writer = MagicMock(return_value=True)
    impl.mla_preprocess_prolog_v3 = MagicMock(return_value=(None, None))
    impl._mla_preprocess = MagicMock(return_value=(None, None, None))
    impl.o_proj = MagicMock(side_effect=lambda tensor, **_kwargs: (tensor,))

    hidden = torch.zeros(3, 128)
    output = torch.empty(3, impl.num_heads * impl.v_head_dim)
    cache = _component_major_cache()
    metadata = SimpleNamespace(
        pcp_prolog=None,
        external_flashmla=None,
        num_actual_tokens=3,
        num_decodes=1,
        num_prefills=2,
        num_decode_tokens=1,
    )

    with (
        patch.object(mla_v1, "_EXTRA_CTX", SimpleNamespace(num_tokens=3)),
        patch.object(mla_v1, "maybe_save_kv_layer_to_connector"),
    ):
        assert impl.forward("layer", hidden, cache, metadata, output) is output

    impl._cache_requires_prolog_v3_writer.assert_called_once_with(cache)
    if expects_prolog:
        impl.mla_preprocess_prolog_v3.assert_called_once_with(hidden, cache, metadata)
        impl._mla_preprocess.assert_not_called()
    else:
        impl.mla_preprocess_prolog_v3.assert_not_called()
        impl._mla_preprocess.assert_called_once()


def test_forward_falls_back_when_dcp_current_buffer_is_too_small():
    impl = _make_impl(AscendMlaDCPImpl)
    impl.use_output_gate = False
    impl.enable_mlapo = False
    impl.use_dcp = True
    impl.use_pcp = False
    impl.is_pcp_decode_sharded = False
    impl._dcp_current_kv_buffers = (
        torch.empty(1, 1, 1, NOPE_DIM),
        torch.empty(1, 1, 1, ROPE_DIM),
        torch.arange(1),
    )
    impl._decode_requires_current_kv = MagicMock(return_value=True)
    impl._cache_requires_prolog_v3_writer = MagicMock(return_value=True)
    impl.mla_preprocess_prolog_v3 = MagicMock(return_value=(None, None))
    impl._mla_preprocess = MagicMock(return_value=(None, None, None))
    impl.o_proj = MagicMock(side_effect=lambda tensor, **_kwargs: (tensor,))

    hidden = torch.zeros(2, 128)
    output = torch.empty(2, impl.num_heads * impl.v_head_dim)
    metadata = SimpleNamespace(
        external_flashmla=None,
        num_actual_tokens=2,
        num_decodes=2,
        num_prefills=0,
        num_decode_tokens=2,
        prefill=None,
    )

    with (
        patch.object(mla_v1, "_EXTRA_CTX", SimpleNamespace(num_tokens=2)),
        patch.object(mla_v1, "maybe_save_kv_layer_to_connector"),
    ):
        assert impl.forward("layer", hidden, _component_major_cache(), metadata, output) is output

    impl.mla_preprocess_prolog_v3.assert_not_called()
    impl._mla_preprocess.assert_called_once()


def test_forward_falls_back_when_dcp_prefill_current_buffer_is_too_small():
    impl = _make_impl(AscendMlaDCPImpl)
    impl.use_output_gate = False
    impl.enable_mlapo = False
    impl.use_dcp = True
    impl.use_pcp = False
    impl.is_pcp_decode_sharded = False
    impl._dcp_current_kv_buffers = (
        torch.empty(1, 1, 1, NOPE_DIM),
        torch.empty(1, 1, 1, ROPE_DIM),
        torch.arange(1),
    )
    impl._dcp_prefill_current_kv_buffers = (
        torch.empty(1, 1, 1, NOPE_DIM),
        torch.empty(1, 1, 1, ROPE_DIM),
        torch.arange(1),
    )
    impl._decode_requires_current_kv = MagicMock(return_value=True)
    impl._prefill_requires_current_kv = MagicMock(return_value=True)
    impl._cache_requires_prolog_v3_writer = MagicMock(return_value=True)
    impl.mla_preprocess_prolog_v3 = MagicMock(return_value=(None, None))
    impl._mla_preprocess = MagicMock(return_value=(None, None, None))
    impl.o_proj = MagicMock(side_effect=lambda tensor, **_kwargs: (tensor,))

    hidden = torch.zeros(2, 128)
    output = torch.empty(2, impl.num_heads * impl.v_head_dim)
    metadata = SimpleNamespace(
        num_actual_tokens=2,
        num_decodes=0,
        num_prefills=1,
        num_decode_tokens=0,
    )

    with (
        patch.object(mla_v1, "_EXTRA_CTX", SimpleNamespace(num_tokens=2)),
        patch.object(mla_v1, "maybe_save_kv_layer_to_connector"),
    ):
        assert impl.forward("layer", hidden, _component_major_cache(), metadata, output) is output

    impl.mla_preprocess_prolog_v3.assert_not_called()
    impl._mla_preprocess.assert_called_once()


@pytest.mark.parametrize("replicate_q", [False, True], ids=["local-heads", "group-heads"])
def test_mla_preprocess_prolog_v3_preserves_dcp_prefill_current_kv(replicate_q):
    impl = _make_impl()
    dcp_impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    dcp_impl.__dict__.update(impl.__dict__)
    impl = dcp_impl
    impl.use_dcp = True
    impl.speculative_config = None
    num_prefill_tokens = 4
    current_nope_cache = torch.zeros(1, BLOCK_SIZE, 1, NOPE_DIM)
    current_pe_cache = torch.zeros(1, BLOCK_SIZE, 1, ROPE_DIM)
    current_indices = torch.arange(BLOCK_SIZE)
    impl._dcp_current_kv_buffers = (
        torch.empty(1, BLOCK_SIZE, 1, NOPE_DIM),
        torch.empty(1, BLOCK_SIZE, 1, ROPE_DIM),
        torch.arange(BLOCK_SIZE),
    )
    impl._dcp_prefill_current_kv_buffers = (current_nope_cache, current_pe_cache, current_indices)
    impl._prepare_prolog_v3_weights = lambda: impl._prolog_v3_weights

    qkv_lora = torch.zeros(num_prefill_tokens, impl.q_lora_rank + FUSED_DIM)
    impl.fused_qkv_a_proj = MagicMock(return_value=(qkv_lora,))
    impl.q_a_layernorm = MagicMock(side_effect=lambda tensor: tensor)
    impl.dcp_q_replicate_enabled = replicate_q
    impl.q_projection_heads = impl.num_heads * (2 if replicate_q else 1)
    # Each projected head and each token has a distinct value. Rank 1 must
    # consume its own heads, without reinterpreting group heads as extra tokens.
    q_rows = 100.0 * torch.arange(num_prefill_tokens)[:, None] + torch.arange(impl.q_projection_heads)
    impl.q_proj = MagicMock(return_value=(q_rows[..., None].expand(-1, -1, impl.qk_head_dim).flatten(1),))
    if replicate_q:
        group = SimpleNamespace(group_size=2, rank_in_group=1)
        impl.q_proj._local_view.side_effect = lambda q: AscendDCPGroupColumnParallelLinear._local_view(group, q)
    impl.rope_single = MagicMock(side_effect=lambda tensor, _cos, _sin: tensor)
    impl.kv_b_proj = MagicMock(
        side_effect=lambda tensor: (
            torch.zeros(tensor.shape[0], impl.num_heads * (impl.qk_nope_head_dim + impl.v_head_dim)),
        )
    )

    nope, rope = _component_major_cache()
    hidden = torch.randn(num_prefill_tokens, 128)
    # Slots -1 are real current rows owned by another DCP rank, not padding.
    prefill_slots = torch.tensor([-1, BLOCK_SIZE, -1, BLOCK_SIZE + 1])
    metadata = SimpleNamespace(
        external_flashmla=None,
        causal=True,
        decode=None,
        num_decode_tokens=0,
        num_actual_tokens=num_prefill_tokens,
        slot_mapping=prefill_slots,
        prefill=SimpleNamespace(
            cos=torch.randn(num_prefill_tokens, ROPE_DIM),
            sin=torch.randn(num_prefill_tokens, ROPE_DIM),
        ),
    )

    expected_nope = torch.arange(num_prefill_tokens, dtype=torch.float32) + 3.0
    expected_pe = torch.arange(num_prefill_tokens, dtype=torch.float32) + 4.0

    def fake_prolog(**kwargs):
        kwargs["kv_cache"][0, :num_prefill_tokens, 0, 0] = expected_nope
        kwargs["kr_cache"][0, :num_prefill_tokens, 0, 0] = expected_pe
        return (
            torch.zeros(num_prefill_tokens, impl.num_heads * NOPE_DIM),
            torch.zeros(num_prefill_tokens, impl.num_heads * ROPE_DIM),
            None,
            None,
            None,
        )

    with (
        patch(
            "vllm_ascend.attention.context_parallel.mla_cp._EXTRA_CTX",
            SimpleNamespace(is_draft_model=False, is_draft_model_prefill=False),
        ),
        patch("torch_npu.npu_mla_prolog_v3", side_effect=fake_prolog) as mock_prolog,
        patch("torch_npu.npu_kv_rmsnorm_rope_cache") as mock_legacy,
        patch.object(mla_v1.DeviceOperator, "reshape_and_cache") as mock_scatter,
    ):
        decode, prefill = impl.mla_preprocess_prolog_v3(hidden, (nope, rope), metadata)

    assert decode is None
    assert prefill is not None
    mock_legacy.assert_not_called()
    expected_q = q_rows[:, -impl.num_heads :]
    torch.testing.assert_close(prefill.q_nope[..., 0], expected_q)
    torch.testing.assert_close(prefill.q_pe[..., 0], expected_q)

    prolog_kwargs = mock_prolog.call_args.kwargs
    assert prolog_kwargs["kv_cache"].untyped_storage() is current_nope_cache.untyped_storage()
    assert prolog_kwargs["kr_cache"].untyped_storage() is current_pe_cache.untyped_storage()
    assert torch.equal(prolog_kwargs["cache_index"], current_indices[:num_prefill_tokens])

    scatter_kwargs = mock_scatter.call_args.kwargs
    assert scatter_kwargs["key_cache"] is nope
    assert scatter_kwargs["value_cache"] is rope
    assert torch.equal(scatter_kwargs["slot_mapping"], prefill_slots)
    assert scatter_kwargs["key"].untyped_storage() is current_nope_cache.untyped_storage()
    assert scatter_kwargs["value"].untyped_storage() is current_pe_cache.untyped_storage()

    nope_passed_to_kv_b = impl.kv_b_proj.call_args.args[0]
    assert torch.equal(nope_passed_to_kv_b[:, 0], expected_nope)
    assert torch.equal(prefill.k_pe[:, 0, 0], expected_pe)


def test_mla_preprocess_prolog_v3_keeps_mixed_dcp_current_kv_disjoint():
    impl = _make_impl()
    dcp_impl = AscendMlaDCPImpl.__new__(AscendMlaDCPImpl)
    dcp_impl.__dict__.update(impl.__dict__)
    impl = dcp_impl
    impl.use_dcp = True
    impl.speculative_config = object()
    num_decode_tokens = 2
    num_prefill_tokens = 4
    decode_nope_cache = torch.zeros(1, BLOCK_SIZE, 1, NOPE_DIM)
    decode_pe_cache = torch.zeros(1, BLOCK_SIZE, 1, ROPE_DIM)
    prefill_nope_cache = torch.zeros(1, BLOCK_SIZE, 1, NOPE_DIM)
    prefill_pe_cache = torch.zeros(1, BLOCK_SIZE, 1, ROPE_DIM)
    current_indices = torch.arange(BLOCK_SIZE)
    impl._dcp_current_kv_buffers = (decode_nope_cache, decode_pe_cache, current_indices)
    impl._dcp_prefill_current_kv_buffers = (prefill_nope_cache, prefill_pe_cache, current_indices)
    impl._prepare_prolog_v3_weights = lambda: impl._prolog_v3_weights
    impl.reorg_decode_q = MagicMock(side_effect=lambda q_nope, q_pe: (q_nope, q_pe))

    total_tokens = num_decode_tokens + num_prefill_tokens
    qkv_lora = torch.zeros(total_tokens, impl.q_lora_rank + FUSED_DIM)
    impl.fused_qkv_a_proj = MagicMock(return_value=(qkv_lora,))
    impl.q_a_layernorm = MagicMock(side_effect=lambda tensor: tensor)
    impl.q_proj = MagicMock(
        side_effect=lambda tensor: (torch.zeros(tensor.shape[0], impl.num_heads * (impl.qk_nope_head_dim + ROPE_DIM)),)
    )
    impl.rope_single = MagicMock(side_effect=lambda tensor, _cos, _sin: tensor)
    impl.kv_b_proj = MagicMock(
        side_effect=lambda tensor: (
            torch.zeros(tensor.shape[0], impl.num_heads * (impl.qk_nope_head_dim + impl.v_head_dim)),
        )
    )

    nope, rope = _component_major_cache()
    hidden = torch.randn(total_tokens, 128)
    metadata = SimpleNamespace(
        external_flashmla=None,
        causal=True,
        decode=SimpleNamespace(
            cos=torch.randn(num_decode_tokens, ROPE_DIM),
            sin=torch.randn(num_decode_tokens, ROPE_DIM),
            actual_seq_lengths_q=[2, 2],
        ),
        prefill=SimpleNamespace(
            cos=torch.randn(num_prefill_tokens, ROPE_DIM),
            sin=torch.randn(num_prefill_tokens, ROPE_DIM),
        ),
        num_decode_tokens=num_decode_tokens,
        num_actual_tokens=total_tokens,
        slot_mapping=torch.tensor([10, 11, -1, BLOCK_SIZE, -1, BLOCK_SIZE + 1]),
    )

    def fake_prolog(**kwargs):
        is_decode = kwargs["kv_cache"].untyped_storage() is decode_nope_cache.untyped_storage()
        kwargs["kv_cache"].fill_(3.0 if is_decode else 5.0)
        kwargs["kr_cache"].fill_(4.0 if is_decode else 6.0)
        return (
            torch.zeros(kwargs["token_x"].shape[0], impl.num_heads * NOPE_DIM),
            torch.zeros(kwargs["token_x"].shape[0], impl.num_heads * ROPE_DIM),
            None,
            None,
            None,
        )

    with (
        patch(
            "vllm_ascend.attention.context_parallel.mla_cp._EXTRA_CTX",
            SimpleNamespace(is_draft_model=False, is_draft_model_prefill=False),
        ),
        patch.object(
            mla_v1,
            "_EXTRA_CTX",
            SimpleNamespace(is_draft_model=False, is_draft_model_prefill=False),
        ),
        patch("torch_npu.npu_mla_prolog_v3", side_effect=fake_prolog),
        patch("torch_npu.npu_kv_rmsnorm_rope_cache") as mock_legacy,
        patch.object(mla_v1.DeviceOperator, "reshape_and_cache") as mock_scatter,
    ):
        decode, prefill = impl.mla_preprocess_prolog_v3(hidden, (nope, rope), metadata)

    mock_legacy.assert_not_called()
    assert decode is not None
    assert prefill is not None
    assert decode.current_k_nope is not None
    assert decode.current_k_pe is not None
    assert torch.all(decode.current_k_nope == 3.0)
    assert torch.all(decode.current_k_pe == 4.0)
    assert torch.all(prefill.k_pe == 6.0)
    assert len(mock_scatter.call_args_list) == 2
    assert mock_scatter.call_args_list[0].kwargs["key"].untyped_storage() is decode_nope_cache.untyped_storage()
    assert mock_scatter.call_args_list[1].kwargs["key"].untyped_storage() is prefill_nope_cache.untyped_storage()


@pytest.mark.parametrize(
    "use_pcp,num_decode_tokens",
    [(False, 1), (True, 1), (True, 2)],
    ids=["speculative-target", "replicated-draft-single", "replicated-draft-multi"],
)
def test_mla_preprocess_prolog_v3_supports_dcp_current_kv_split(use_pcp, num_decode_tokens):
    impl = _make_impl(AscendMlaDCPImpl)
    impl.use_pcp = use_pcp
    impl.use_dcp = True
    impl.speculative_config = object()
    # A target of speculative decoding always splits. A replicated draft only
    # splits a multi-token query; its single-token step writes history directly.
    requires_current = not use_pcp or num_decode_tokens > 1
    # Two draft tokens span two scratch pages, but fit inside one history page.
    scratch_pages, scratch_page_size = (2, 1) if num_decode_tokens > 1 else (1, BLOCK_SIZE)
    current_nope_cache = torch.zeros(scratch_pages, scratch_page_size, 1, NOPE_DIM)
    current_pe_cache = torch.zeros(scratch_pages, scratch_page_size, 1, ROPE_DIM)
    current_indices = torch.arange(scratch_pages * scratch_page_size)
    impl._dcp_current_kv_buffers = (current_nope_cache, current_pe_cache, current_indices)
    events = []

    def gather_q(*qs, dim):
        events.append("dcp_q")
        assert dim == 1
        return qs

    impl._dcp_all_gather_fragments = MagicMock(side_effect=gather_q)

    nope, rope = _component_major_cache()
    nope.zero_()
    rope.zero_()
    hidden = torch.randn(num_decode_tokens, 128)
    slots = torch.tensor([10, -1][:num_decode_tokens])
    metadata = SimpleNamespace(
        pcp_prolog=None,
        external_flashmla=None,
        causal=True,
        num_decode_tokens=num_decode_tokens,
        num_actual_tokens=num_decode_tokens,
        slot_mapping=slots,
        decode=SimpleNamespace(
            actual_seq_lengths_q=[num_decode_tokens],
            cos=torch.randn(num_decode_tokens, ROPE_DIM),
            sin=torch.randn(num_decode_tokens, ROPE_DIM),
        ),
        prefill=None,
    )

    def fake_prolog(**kwargs):
        events.append("prolog")
        page_size = kwargs["kv_cache"].shape[1]
        for row, slot in enumerate(kwargs["cache_index"].tolist()):
            if slot >= 0:
                page, offset = divmod(slot, page_size)
                kwargs["kv_cache"][page, offset].fill_(3.0 + row)
                kwargs["kr_cache"][page, offset].fill_(4.0 + row)
        return (
            torch.zeros(num_decode_tokens, impl.num_heads * NOPE_DIM),
            torch.zeros(num_decode_tokens, impl.num_heads * ROPE_DIM),
            None,
            None,
            None,
        )

    with (
        patch(
            "vllm_ascend.attention.context_parallel.mla_cp._EXTRA_CTX",
            SimpleNamespace(is_draft_model=use_pcp, is_draft_model_prefill=False),
        ),
        patch.object(mla_v1, "get_pcp_group") as pcp_group,
        patch("torch_npu.npu_mla_prolog_v3", side_effect=fake_prolog) as mock_prolog,
        patch("torch_npu.npu_kv_rmsnorm_rope_cache") as mock_legacy,
        patch.object(
            mla_v1.DeviceOperator, "reshape_and_cache", side_effect=lambda **_: events.append("scatter")
        ) as mock_scatter,
    ):
        decode, prefill = impl.mla_preprocess_prolog_v3(hidden, (nope, rope), metadata)

    mock_prolog.assert_called_once()
    assert decode is not None and prefill is None
    prolog_kwargs = mock_prolog.call_args.kwargs
    torch.testing.assert_close(prolog_kwargs["token_x"], hidden)
    torch.testing.assert_close(prolog_kwargs["rope_cos"], metadata.decode.cos)
    torch.testing.assert_close(prolog_kwargs["rope_sin"], metadata.decode.sin)
    mock_legacy.assert_not_called()
    pcp_group.assert_not_called()
    impl._dcp_all_gather_fragments.assert_called_once()
    # Without a PCP plan, preserve the native DCP order: reorg Q before history scatter.
    assert events == (["prolog", "dcp_q", "scatter"] if requires_current else ["prolog", "dcp_q"])
    if requires_current:
        assert prolog_kwargs["kv_cache"].untyped_storage() is current_nope_cache.untyped_storage()
        assert prolog_kwargs["kr_cache"].untyped_storage() is current_pe_cache.untyped_storage()
        torch.testing.assert_close(prolog_kwargs["cache_index"], current_indices[:num_decode_tokens])
        torch.testing.assert_close(decode.current_k_nope[:, 0, 0], 3.0 + torch.arange(num_decode_tokens))
        torch.testing.assert_close(decode.current_k_pe[:, 0, 0], 4.0 + torch.arange(num_decode_tokens))
        mock_scatter.assert_called_once()
        assert mock_scatter.call_args.kwargs["key_cache"] is nope
        assert mock_scatter.call_args.kwargs["value_cache"] is rope
        torch.testing.assert_close(mock_scatter.call_args.kwargs["slot_mapping"], slots)
    else:
        assert prolog_kwargs["kv_cache"] is nope and prolog_kwargs["kr_cache"] is rope
        torch.testing.assert_close(prolog_kwargs["cache_index"], slots)
        assert decode.current_k_nope is None and decode.current_k_pe is None
        assert nope[0, 10, 0, 0] == 3.0 and rope[0, 10, 0, 0] == 4.0
        mock_scatter.assert_not_called()


def test_mla_preprocess_prolog_v3_keeps_prefill_query_unquantized_for_fa_quant():
    impl = _make_impl()
    total_tokens = 3
    qkv_lora = torch.zeros(total_tokens, impl.q_lora_rank + FUSED_DIM)
    impl.fused_qkv_a_proj = MagicMock(return_value=(qkv_lora,))
    impl.q_a_layernorm = MagicMock(side_effect=lambda tensor: tensor)
    impl.q_proj = MagicMock(
        side_effect=lambda tensor: (torch.zeros(tensor.shape[0], impl.num_heads * (impl.qk_nope_head_dim + ROPE_DIM)),)
    )
    impl.rope_single = MagicMock(side_effect=lambda tensor, _cos, _sin: tensor)
    impl.fa_quant_layer = True
    impl.support_fp8_attention = True
    impl.quant_kscale = torch.tensor([[2.0]])
    impl.fak_descale_reciprocal = torch.tensor(4.0)
    impl.fak_descale_float = torch.tensor(0.25)
    nope, rope = _token_fused_cache()

    num_decode = 1
    num_prefill = 2
    hidden = torch.randn(num_decode + num_prefill, 128)
    metadata = SimpleNamespace(
        external_flashmla=None,
        num_decode_tokens=num_decode,
        num_actual_tokens=num_decode + num_prefill,
        slot_mapping=torch.tensor([0, 128, 129]),
        decode=SimpleNamespace(
            cos=torch.randn(num_decode, ROPE_DIM),
            sin=torch.randn(num_decode, ROPE_DIM),
        ),
        prefill=SimpleNamespace(
            cos=torch.randn(num_prefill, ROPE_DIM),
            sin=torch.randn(num_prefill, ROPE_DIM),
        ),
    )
    impl.kv_b_proj.return_value = (
        torch.zeros(num_prefill, impl.num_heads * (impl.qk_nope_head_dim + impl.v_head_dim)),
    )
    impl._prepare_prolog_v3_weights = lambda: impl._prolog_v3_weights

    def fake_prolog(**kwargs):
        tokens = kwargs["token_x"].shape[0]
        return (
            torch.zeros(tokens, impl.num_heads * NOPE_DIM),
            torch.zeros(tokens, impl.num_heads * ROPE_DIM),
            torch.ones(tokens, impl.num_heads),
            None,
            None,
        )

    with (
        patch("torch_npu.npu_mla_prolog_v3", side_effect=fake_prolog) as mock_prolog,
        patch.object(mla_v1, "load_mla_cache_rows_by_slots", side_effect=_load_cache_rows_on_cpu),
    ):
        impl.mla_preprocess_prolog_v3(hidden, (nope, rope), metadata)

    assert mock_prolog.call_count == 2
    decode_kwargs = mock_prolog.call_args_list[0].kwargs
    prefill_kwargs = mock_prolog.call_args_list[1].kwargs
    assert decode_kwargs["query_quant_mode"] == 1
    assert prefill_kwargs["query_quant_mode"] == 0
    assert decode_kwargs["kv_cache_quant_mode"] == prefill_kwargs["kv_cache_quant_mode"] == 1
    assert decode_kwargs["cache_mode"] == prefill_kwargs["cache_mode"] == "PA_BSND"
    assert decode_kwargs["tile_size"] == prefill_kwargs["tile_size"] == BLOCK_SIZE
    kv_b_proj_input = impl.kv_b_proj.call_args.args[0]
    assert kv_b_proj_input.dtype == nope.dtype
    torch.testing.assert_close(kv_b_proj_input, nope[1, :num_prefill, 0] * impl.fak_descale_float)
    assert torch.equal(decode_kwargs["quant_scale_ckv"], impl.fak_descale_reciprocal)


def _make_wide_draft_impl():
    impl = _make_impl()
    impl.fused_qkv_a_proj = SimpleNamespace(input_size=14336, weight=torch.empty(1))
    impl.support_fp8_attention = True
    impl.is_draft_model = True
    impl.use_pcp = False
    impl.use_dcp = False
    impl.is_pcp_decode_sharded = False
    impl.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=257),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[128, 384]),
        cache_config=SimpleNamespace(block_size=BLOCK_SIZE),
    )
    impl.kv_a_layernorm = SimpleNamespace(weight=torch.ones(NOPE_DIM), variance_epsilon=1e-6)
    impl._strided_kv_workspace = None
    return impl


@pytest.mark.parametrize("input_size", [7168, 8192, 14336, None])
def test_prolog_writer_checks_projection_input_instead_of_model_hidden_size(input_size):
    impl = _make_wide_draft_impl()
    impl.fused_qkv_a_proj.input_size = input_size
    assert impl.supports_prolog_v3_writer() is (input_size in (7168, 8192))


@pytest.mark.parametrize("excluded", ["use_pcp", "use_dcp", "enable_kv_nz", "fa_quant_layer"])
def test_wide_draft_adapter_preserves_existing_exclusions(excluded):
    impl = _make_wide_draft_impl()
    setattr(impl, excluded, True)
    impl._prepare_strided_kv_workspace()
    assert not impl.supports_strided_mla_writer()
    assert impl._strided_kv_workspace is None


@pytest.mark.parametrize(
    "attribute,value", [("support_fp8_attention", False), ("is_draft_model", False), ("dtype", torch.float16)]
)
def test_wide_draft_adapter_does_not_expand_hardware_or_dtype_support(attribute, value):
    impl = _make_wide_draft_impl()
    setattr(impl, attribute, value)
    impl._prepare_strided_kv_workspace()
    assert not impl.supports_strided_mla_writer()
    assert impl._strided_kv_workspace is None


def test_wide_draft_workspace_is_bounded_prepared_and_stable_across_reload():
    impl = _make_wide_draft_impl()
    assert not impl.supports_strided_mla_writer()
    impl._prepare_strided_kv_workspace()
    assert impl.supports_strided_mla_writer()
    workspace = impl._strided_kv_workspace
    assert workspace is not None
    assert workspace[0].shape == (3, BLOCK_SIZE, 1, NOPE_DIM)
    assert workspace[1].shape == (3, BLOCK_SIZE, 1, ROPE_DIM)
    assert torch.equal(workspace[2], torch.arange(384))
    assert sum(t.numel() * t.element_size() for t in workspace) == 384 * (FUSED_DIM * 2 + 8)
    impl._prepare_strided_kv_workspace()
    assert impl._strided_kv_workspace is workspace


def test_wide_draft_weight_loading_prepares_workspace_without_prolog_weights():
    impl = _make_wide_draft_impl()
    impl.enable_mlapo = False
    impl._prolog_v3_weights = None
    impl.kv_b_proj = SimpleNamespace(
        quant_method=mla_v1.UnquantizedLinearMethod(),
        weight=torch.empty(impl.num_heads * (impl.qk_nope_head_dim + impl.v_head_dim), NOPE_DIM),
    )
    with (
        patch("torch_npu.npu_format_cast", side_effect=lambda tensor, _format: tensor),
        patch.object(mla_v1, "maybe_trans_nz", side_effect=lambda tensor: tensor),
        patch.object(impl, "_prepare_prolog_v3_weights") as prepare_prolog,
    ):
        impl.process_weights_after_loading(torch.bfloat16)
        assert impl.supports_strided_mla_writer()
        workspace = impl._strided_kv_workspace
        impl.process_weights_after_loading(torch.bfloat16)
        assert impl._strided_kv_workspace is workspace
        prepare_prolog.assert_not_called()
    assert impl._prolog_v3_weights is None


@pytest.mark.parametrize("cache_factory", [_component_major_cache, _token_fused_cache])
@pytest.mark.parametrize("prefill", [False, True])
def test_wide_draft_adapter_writes_original_storage_and_skips_padding(cache_factory, prefill):
    impl = _make_wide_draft_impl()
    impl._prepare_strided_kv_workspace()
    cache = cache_factory()
    base = torch.empty(0, dtype=cache[0].dtype).set_(
        cache[0].untyped_storage(), 0, (cache[0].untyped_storage().nbytes() // cache[0].element_size(),), (1,)
    )
    expected = base.clone()
    slots = torch.tensor([0, BLOCK_SIZE - 1, BLOCK_SIZE, -1])
    kv = torch.randn(4, FUSED_DIM, dtype=torch.bfloat16)
    cos, sin = torch.randn(4, ROPE_DIM), torch.randn(4, ROPE_DIM)
    normalized = kv[:, :NOPE_DIM].float()
    normalized = normalized * torch.rsqrt(normalized.square().mean(-1, keepdim=True) + 1e-6)
    normalized = normalized.to(kv.dtype).unsqueeze(1)
    raw_rope = kv[:, NOPE_DIM:].float()
    rotated = torch.stack((-raw_rope[:, 1::2], raw_rope[:, ::2]), dim=-1).flatten(1)
    rotated = (raw_rope * cos + rotated * sin).to(kv.dtype).unsqueeze(1)
    for component, values in zip(cache, (normalized, rotated)):
        target = expected.as_strided(component.shape, component.stride(), component.storage_offset())
        for token, slot in enumerate(slots.tolist()):
            if slot >= 0:
                target[slot // BLOCK_SIZE, slot % BLOCK_SIZE] = values[token]

    def normalize_into_temporary(_kv, _gamma, _cos, _sin, indices, rope, nope, **kwargs):
        assert rope.is_contiguous() and nope.is_contiguous()
        assert rope.untyped_storage() is not cache[0].untyped_storage()
        assert nope.untyped_storage() is not cache[0].untyped_storage()
        assert torch.equal(indices, torch.arange(4))
        assert kwargs["is_output_kv"] is False
        assert kwargs["cache_mode"] == "PA"
        nope.flatten(0, 1)[:4].copy_(normalized)
        rope.flatten(0, 1)[:4].copy_(rotated)

    def scatter(**kwargs):
        assert kwargs["key_cache"] is cache[0]
        assert kwargs["value_cache"] is cache[1]
        for token, slot in enumerate(kwargs["slot_mapping"].tolist()):
            if slot >= 0:
                kwargs["key_cache"][slot // BLOCK_SIZE, slot % BLOCK_SIZE] = kwargs["key"][token]
                kwargs["value_cache"][slot // BLOCK_SIZE, slot % BLOCK_SIZE] = kwargs["value"][token]

    with (
        patch("torch_npu.npu_kv_rmsnorm_rope_cache", side_effect=normalize_into_temporary),
        patch("torch_npu.npu_mla_prolog_v3") as prolog,
        patch.object(mla_v1.DeviceOperator, "reshape_and_cache", side_effect=scatter),
    ):
        if prefill:
            rope, nope = impl.exec_kv_prefill(kv, cos, sin, cache, slots)
        else:
            history_rope, history_nope, rope, nope = impl.exec_kv_decode(
                kv, cos, sin, cache, slots, return_current_kv=True
            )
            assert history_nope is cache[0] and history_rope is cache[1]
        prolog.assert_not_called()
    torch.testing.assert_close(nope, normalized)
    torch.testing.assert_close(rope, rotated)
    torch.testing.assert_close(base, expected)


def test_wide_draft_workspace_never_grows_during_forward():
    impl = _make_wide_draft_impl()
    cache = _token_fused_cache()
    with pytest.raises(RuntimeError, match="prepared before KV cache sizing"):
        impl._exec_strided_kv_writer(torch.empty(1, FUSED_DIM), None, None, cache, torch.tensor([0]))
    impl._prepare_strided_kv_workspace()
    workspace = impl._strided_kv_workspace
    with pytest.raises(RuntimeError, match="exceeds the profiled workspace"):
        impl._exec_strided_kv_writer(torch.empty(385, FUSED_DIM), None, None, cache, torch.arange(385))
    with (
        patch("torch_npu.npu_kv_rmsnorm_rope_cache") as writer,
        patch.object(mla_v1.DeviceOperator, "reshape_and_cache") as scatter,
    ):
        rope, nope = impl._exec_strided_kv_writer(torch.empty(0, FUSED_DIM), None, None, cache, torch.empty(0))
        assert rope.shape == (0, 1, ROPE_DIM) and nope.shape == (0, 1, NOPE_DIM)
        writer.assert_not_called()
        scatter.assert_not_called()
    assert impl._strided_kv_workspace is workspace


def test_wide_draft_forward_uses_regular_projection_for_fused_cache():
    impl = _make_wide_draft_impl()
    impl._prepare_strided_kv_workspace()
    impl.use_output_gate = False
    impl.enable_mlapo = True  # The adapter must take priority even over the older decode fusion.
    impl.mla_preprocess_prolog_v3 = MagicMock(return_value=(None, None))
    impl.mla_preprocess_only_decode = MagicMock(return_value=(None, None))
    impl._mla_preprocess = MagicMock(return_value=(None, None, None))
    impl.o_proj = MagicMock(side_effect=lambda tensor, **_kwargs: (tensor,))
    hidden = torch.zeros(1, 14336)
    output = torch.empty(1, impl.num_heads * impl.v_head_dim)
    cache = torch.zeros(2, BLOCK_SIZE, 1, FUSED_DIM)
    metadata = SimpleNamespace(num_actual_tokens=1, num_decodes=1, num_prefills=0, num_decode_tokens=1)
    with (
        patch.object(mla_v1, "_EXTRA_CTX", SimpleNamespace(num_tokens=1)),
        patch.object(mla_v1, "maybe_save_kv_layer_to_connector"),
    ):
        assert impl.forward("layer", hidden, cache, metadata, output) is output
    impl._mla_preprocess.assert_called_once()
    impl.mla_preprocess_prolog_v3.assert_not_called()
    impl.mla_preprocess_only_decode.assert_not_called()


@pytest.mark.parametrize("use_v2_runner", [False, True])
def test_dense_writer_prepares_weights_with_sparse_target(monkeypatch, use_v2_runner):
    impl = _make_post_load_impl(monkeypatch)
    impl.vllm_config.use_v2_model_runner = use_v2_runner
    impl.vllm_config.parallel_config.decode_context_parallel_size = 1
    impl.vllm_config.model_config.hf_text_config.index_topk = 2048
    impl.use_flashmla = True
    impl._flashmla_writer_buffers = None
    assert enable_sfa(impl.vllm_config)

    impl.process_weights_after_loading(torch.bfloat16)

    assert impl._prolog_v3_weights is not None


@pytest.mark.parametrize("use_v2_runner", [False, True])
def test_fia_draft_skips_unused_prolog_weights_with_sparse_target(monkeypatch, use_v2_runner):
    impl = _make_post_load_impl(monkeypatch)
    config = impl.vllm_config
    config.use_v2_model_runner = use_v2_runner
    config.model_config.hf_text_config.index_topk = 2048
    config.parallel_config.decode_context_parallel_size = 1
    assert enable_sfa(config)
    assert not impl.use_flashmla and not impl.enable_mlapo
    cache = (torch.empty(2, BLOCK_SIZE, 1, 4), torch.empty(2, BLOCK_SIZE, 1, 2))
    assert not impl._cache_requires_prolog_v3_writer(cache)

    impl.process_weights_after_loading(torch.bfloat16)

    assert impl._prolog_v3_weights is None


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize(
    "sharded,local_prefill,global_prefill,pcp_layout,expected_plan",
    [
        (True, False, False, True, True),
        (False, True, True, True, True),
        (False, False, True, True, True),
        (False, False, False, True, False),
        (True, False, False, False, False),
    ],
)
def test_pcp_prolog_metadata_uses_global_batch_layout(
    rank, sharded, local_prefill, global_prefill, pcp_layout, expected_plan
):
    """An empty local prefill still joins, while replicated draft steps do not."""
    builder = mla_v1.AscendMLAMetadataBuilder.__new__(mla_v1.AscendMLAMetadataBuilder)
    builder.use_mla_rope = True
    builder.rope_dim = ROPE_DIM
    builder.is_pcp_decode_sharded = sharded
    builder._pcp_prolog_layers = (
        SimpleNamespace(impl=SimpleNamespace(_uses_pcp_prolog_v3=lambda _cache: True), kv_cache=()),
    )
    slots = torch.tensor([[5, 7, -1], [-1, 9, -1]])
    original_slots = slots.clone()
    metadata = SimpleNamespace(
        pcp_prolog=None,
        uses_pcp_sharded_layout=pcp_layout,
        num_decode_tokens=1,
        num_prefills=int(local_prefill),
        slot_mapping=slots.flatten(),
    )
    common = SimpleNamespace(
        pcp_context=SimpleNamespace(global_batch=SimpleNamespace(is_prefilling_np=torch.tensor([global_prefill])))
        if pcp_layout
        else None
    )
    group = SimpleNamespace(world_size=2, rank_in_group=rank)
    with patch.object(mla_v1, "get_pcp_group", return_value=group):
        builder._build_pcp_prolog_metadata(metadata, common)

    assert (metadata.pcp_prolog is not None) is expected_plan
    torch.testing.assert_close(slots, original_slots)
    if expected_plan:
        replicated = 0 if sharded else 1
        expected_local = slots[rank].clone()
        expected_local[:replicated] = slots[0, :replicated]
        torch.testing.assert_close(metadata.pcp_prolog.local_slots, expected_local)
        torch.testing.assert_close(metadata.pcp_prolog.global_slots, slots[:, replicated:].flatten())
        assert metadata.pcp_prolog.num_replicated_tokens == replicated


def test_pcp_prolog_metadata_follows_layer_cache_binding():
    impl = _make_impl()
    impl.use_pcp = impl._pcp_prolog_uses_split_history = True
    impl._pcp_prolog_v3_buffers = (torch.empty(0), torch.empty(0), torch.arange(3))
    layer = SimpleNamespace(impl=impl, kv_cache=torch.empty(0))
    builder = mla_v1.AscendMLAMetadataBuilder.__new__(mla_v1.AscendMLAMetadataBuilder)
    builder.use_mla_rope = True
    builder.rope_dim = ROPE_DIM
    builder.is_pcp_decode_sharded = False
    builder._pcp_prolog_layers = (layer,)
    common = SimpleNamespace(
        pcp_context=SimpleNamespace(global_batch=SimpleNamespace(is_prefilling_np=torch.tensor([True])))
    )
    strided_cache = _component_major_cache(dtype=torch.bfloat16)
    split_cache = tuple(component.contiguous() for component in strided_cache)
    unsupported_cache = tuple(component.to(torch.float16) for component in split_cache)
    with patch.object(mla_v1, "get_pcp_group", return_value=SimpleNamespace(world_size=2, rank_in_group=0)) as group:
        for cache, expected in (
            (torch.empty(0), False),  # Metadata builders exist before KV binding.
            (split_cache, True),
            (strided_cache, True),
            (unsupported_cache, False),  # Rebinding must not reuse a stale decision.
        ):
            layer.kv_cache = cache
            metadata = SimpleNamespace(
                pcp_prolog=None,
                uses_pcp_sharded_layout=True,
                num_decode_tokens=1,
                num_prefills=0,
                slot_mapping=torch.arange(6),
            )
            builder._build_pcp_prolog_metadata(metadata, common)
            assert (metadata.pcp_prolog is not None) is expected
        assert group.call_count == 2


@pytest.mark.parametrize("writer_ready", [False, True])
@pytest.mark.parametrize("sharded_decode", [False, True])
def test_pcp_builder_refreshes_padded_rope_only_for_consumers(monkeypatch, writer_ready, sharded_decode):
    """Empty ranks retain the writer and native sharded-decode RoPE contracts."""
    builder = mla_v1.AscendMLAMetadataBuilder.__new__(mla_v1.AscendMLAMetadataBuilder)
    builder.use_pcp = builder.use_dcp = builder.use_mla_rope = True
    builder.is_pcp_decode_sharded = sharded_decode
    consumer = SimpleNamespace(
        impl=SimpleNamespace(_uses_pcp_prolog_v3=MagicMock(return_value=writer_ready)), kv_cache=()
    )
    builder._pcp_prolog_layers = (consumer,)
    builder.pcp_size = 2
    builder.rope_dim = ROPE_DIM
    builder.flashmla_state = None
    builder.decode_threshold = 1
    builder.metadata_cls = mla_v1.AscendMLAMetadata
    builder.model_config = SimpleNamespace(get_head_size=lambda: FUSED_DIM)
    builder.vllm_config = SimpleNamespace()
    builder.attn_mask_builder = SimpleNamespace(get_splitfuse_attn_mask=lambda: None)
    positions = torch.tensor([4, 7, 0])
    common = SimpleNamespace(
        pcp_context=SimpleNamespace(global_batch=SimpleNamespace(is_prefilling_np=torch.tensor([True]))),
        pcp_cache_group_idx=0,
        slot_mapping=torch.tensor([1, 2, 3, -1, -1, -1]),
        positions=positions,
        num_reqs=0,
        num_actual_tokens=0,
        num_input_tokens=positions.numel(),
        query_start_loc=torch.tensor([0]),
        query_start_loc_cpu=torch.tensor([0]),
        context_parallel_metadata=None,
        max_query_len=0,
        _seq_lens_cpu=torch.empty(0, dtype=torch.int32),
        graph_pad_size=-1,
        block_table_tensor=torch.empty((0, 1), dtype=torch.int32),
        attn_state=mla_v1.AscendAttentionState.ChunkedPrefill,
        causal=True,
    )
    table = torch.arange(16, dtype=torch.float32)[:, None].expand(-1, ROPE_DIM).contiguous()
    cos_buffer = torch.zeros(3, 1, 1, ROPE_DIM)
    sin_buffer = torch.zeros_like(cos_buffer)
    monkeypatch.setattr(rotary_embedding, "_cos_sin_flat", None)
    monkeypatch.setattr(rotary_embedding, "_cos_cache", table)
    monkeypatch.setattr(rotary_embedding, "_sin_cache", -table)
    monkeypatch.setattr(rotary_embedding, "_cos_mla", cos_buffer)
    monkeypatch.setattr(rotary_embedding, "_sin_mla", sin_buffer)
    with (
        patch.object(mla_v1, "get_pcp_group", return_value=SimpleNamespace(world_size=2, rank_in_group=1)),
        patch.object(mla_v1, "is_pd_decode_recompute_scheduler_enabled", return_value=False),
        patch.object(mla_v1, "get_cos_and_sin_mla", wraps=rotary_embedding.get_cos_and_sin_mla) as get_rope,
    ):
        metadata = builder.build(0, common)
        consumer.impl._uses_pcp_prolog_v3.assert_called_once_with(consumer.kv_cache)
        assert (metadata.pcp_prolog is not None) is writer_ready
        if not writer_ready and not sharded_decode:
            get_rope.assert_not_called()
            assert metadata.pcp_cos is metadata.pcp_sin is None
            assert torch.count_nonzero(cos_buffer) == torch.count_nonzero(sin_buffer) == 0
            return
        get_rope.assert_called_once()
        torch.testing.assert_close(get_rope.call_args.args[0], positions)
        assert get_rope.call_args.kwargs == {"use_cache": True}
        assert metadata.num_actual_tokens == 0 and metadata.prefill is None and metadata.decode is None
        if writer_ready:
            torch.testing.assert_close(metadata.pcp_prolog.local_slots, torch.full((3,), -1))
        assert metadata.pcp_cos.shape == (3, 1, 1, ROPE_DIM)
        assert metadata.pcp_cos.data_ptr() == cos_buffer.data_ptr()
        assert metadata.pcp_sin.data_ptr() == sin_buffer.data_ptr()
        torch.testing.assert_close(metadata.pcp_cos[:, 0, 0, 0], positions.float())
        torch.testing.assert_close(metadata.pcp_sin[:, 0, 0, 0], -positions.float())

        common.positions = torch.tensor([6, 8, 0])
        refreshed = builder.build(0, common)
        assert refreshed.pcp_cos.data_ptr() == metadata.pcp_cos.data_ptr()
        torch.testing.assert_close(refreshed.pcp_cos[:, 0, 0, 0], common.positions.float())


@pytest.mark.parametrize("impl_type", [AscendMLAImpl, AscendMlaDCPImpl])
@pytest.mark.parametrize("block_size", [BLOCK_SIZE, 1536])
def test_pcp_post_load_profiles_workspace_and_preserves_addresses_on_reload(monkeypatch, impl_type, block_size):
    impl = _make_post_load_impl(monkeypatch, impl_type)
    for projection in (impl.fused_qkv_a_proj, impl.q_proj):
        projection.weight = projection.weight.T.contiguous()
        projection.quant_method = UnquantizedLinearMethod()
    impl.vllm_config.cache_config.block_size = block_size
    impl.use_pcp = True
    impl.is_pcp_decode_sharded = not impl.use_dcp
    impl.vllm_config.parallel_config.prefill_context_parallel_size = 2
    impl.vllm_config.compilation_config.cudagraph_capture_sizes = [8, 32]
    impl.vllm_config.use_v2_model_runner = True
    owner = impl.vllm_config.compilation_config.static_forward_context[impl.layer_name]
    cache_spec = owner.get_kv_cache_spec(impl.vllm_config)

    impl.process_weights_after_loading(torch.bfloat16)

    buffers = impl._pcp_prolog_v3_buffers
    prepared = impl._prolog_v3_weights
    assert buffers is not None and prepared is not None
    assert buffers[2].numel() == 32
    assert buffers[0].shape[1] == buffers[1].shape[1] == BLOCK_SIZE
    assert buffers[0].flatten(0, 1).shape[0] >= 32
    assert impl._pcp_prolog_v3_buffers_are_sufficient()
    assert supports_strided_mla_cache(impl.vllm_config, owner, cache_spec)
    pointers = [tensor.data_ptr() for tensor in buffers]
    weight_pointer = prepared["weight_dq"].data_ptr()
    impl.fused_qkv_a_proj.weight.fill_(2)
    impl.process_weights_after_loading(torch.bfloat16)

    assert impl._pcp_prolog_v3_buffers is buffers
    assert [tensor.data_ptr() for tensor in buffers] == pointers
    assert impl._prolog_v3_weights is prepared
    assert prepared["weight_dq"].data_ptr() == weight_pointer
    torch.testing.assert_close(prepared["weight_dq"], torch.full((4, 4), 2.0))
    impl._pcp_prolog_v3_buffers = None
    assert not supports_strided_mla_cache(impl.vllm_config, owner, cache_spec)


@pytest.mark.parametrize("impl_type", [AscendMLAImpl, AscendMlaDCPImpl])
def test_pcp_static_w8a8_does_not_prepare_a_prolog_writer(monkeypatch, impl_type):
    impl = _make_post_load_impl(monkeypatch, impl_type)
    impl.use_pcp = True
    impl.is_pcp_decode_sharded = not impl.use_dcp
    impl.vllm_config.parallel_config.prefill_context_parallel_size = 2
    impl.vllm_config.use_v2_model_runner = True
    owner = impl.vllm_config.compilation_config.static_forward_context[impl.layer_name]

    impl.process_weights_after_loading(torch.bfloat16)

    assert impl._pcp_prolog_v3_buffers is None
    assert impl._prolog_v3_weights is None
    assert impl._dcp_current_kv_buffers is None
    assert impl._dcp_prefill_current_kv_buffers is None
    assert not supports_strided_mla_cache(impl.vllm_config, owner, owner.get_kv_cache_spec(impl.vllm_config))


@pytest.mark.parametrize(
    "connector,layerwise,module_path,expected",
    [
        (None, False, None, True),
        ("MooncakeConnectorV2", False, None, True),
        ("MooncakePullConnector", False, None, True),
        ("UnsupportedConnector", False, None, False),
        ("MooncakeConnectorV2", True, None, False),
        ("MooncakeConnectorV2", False, "custom.connector", False),
    ],
)
def test_pcp_post_load_allocates_only_for_supported_connectors(
    monkeypatch, connector, layerwise, module_path, expected
):
    impl = _make_post_load_impl(monkeypatch)
    impl.use_pcp = True
    impl.vllm_config.parallel_config.prefill_context_parallel_size = 2
    impl.vllm_config.use_v2_model_runner = True
    if connector is not None:
        impl.vllm_config.kv_transfer_config = SimpleNamespace(
            kv_connector=connector,
            kv_connector_module_path=module_path,
            kv_connector_extra_config={"use_layerwise": layerwise},
        )
    for projection in (impl.fused_qkv_a_proj, impl.q_proj):
        projection.quant_method.quant_method = object.__new__(AscendW8A8DynamicLinearMethod)

    impl.process_weights_after_loading(torch.bfloat16)

    assert (impl._pcp_prolog_v3_buffers is not None) is expected
    assert (impl._prolog_v3_weights is not None) is expected


@pytest.mark.parametrize("use_dcp", [False, True])
@pytest.mark.parametrize("capacity,expected", [(0, False), (2, False), (3, True)])
def test_pcp_forward_requires_prepared_capacity(use_dcp, capacity, expected):
    impl = _make_impl()
    impl.use_pcp = True
    impl.use_dcp = use_dcp
    impl.use_output_gate = False
    impl.enable_mlapo = False
    if capacity:
        impl._pcp_prolog_v3_buffers = (torch.empty(0), torch.empty(0), torch.arange(capacity))
    impl._cache_requires_prolog_v3_writer = MagicMock(return_value=True)
    impl.mla_preprocess_prolog_v3 = MagicMock(return_value=(None, None))
    impl._mla_preprocess = MagicMock(return_value=(None, None, None))
    impl.o_proj = MagicMock(side_effect=lambda tensor, **_kwargs: (tensor,))
    metadata = SimpleNamespace(
        external_flashmla=None,
        pcp_prolog=mla_v1.PCPMLAPrologMetadata(torch.arange(3), torch.arange(6), 0),
        num_actual_tokens=3,
        num_decodes=1,
        num_prefills=1,
        num_decode_tokens=1,
    )
    hidden = torch.zeros(3, 128)
    output = torch.empty(3, impl.num_heads * impl.v_head_dim)
    with (
        patch.object(mla_v1, "_EXTRA_CTX", SimpleNamespace(num_tokens=3)),
        patch.object(mla_v1, "maybe_save_kv_layer_to_connector"),
    ):
        assert impl.forward("layer", hidden, _component_major_cache(), metadata, output) is output

    assert impl.mla_preprocess_prolog_v3.called is expected
    assert impl._mla_preprocess.called is (not expected)


@pytest.mark.parametrize(
    "case,expect_prolog",
    [
        ("mixed", True),
        ("padding_only_rank", True),
        ("decode_without_plan", False),
        ("layer_not_selected", False),
        ("short_workspace", False),
        ("wrong_dtype", False),
        ("wrong_component_width", False),
        ("mismatched_pages", False),
        ("static_w8a8", False),
    ],
)
def test_pcp_forward_with_split_history(case, expect_prolog):
    """A contiguous cache must neither disable a ready PCP writer nor enable one on its own."""
    impl = _make_impl()
    impl.use_pcp = impl.is_pcp_decode_sharded = True
    impl.enable_mlapo = False
    if case == "static_w8a8":
        for projection in (impl.fused_qkv_a_proj, impl.q_proj):
            projection.quant_method = SimpleNamespace(quant_method=object.__new__(AscendW8A8LinearMethod))
    impl._pcp_prolog_uses_split_history = case != "layer_not_selected"
    capacity = 2 if case == "short_workspace" else 3
    impl._pcp_prolog_v3_buffers = (
        torch.empty(1, BLOCK_SIZE, 1, NOPE_DIM, dtype=torch.bfloat16),
        torch.empty(1, BLOCK_SIZE, 1, ROPE_DIM, dtype=torch.bfloat16),
        torch.arange(capacity),
    )
    dtype = torch.float16 if case == "wrong_dtype" else torch.bfloat16
    rope_pages = 1 if case == "mismatched_pages" else 2
    rope_dim = ROPE_DIM + 1 if case == "wrong_component_width" else ROPE_DIM
    history = (
        torch.empty(2, BLOCK_SIZE, 1, NOPE_DIM, dtype=dtype),
        torch.empty(rope_pages, BLOCK_SIZE, 1, rope_dim, dtype=dtype),
    )
    assert not impl._cache_requires_prolog_v3_writer(history)
    decode_only = case == "decode_without_plan"
    metadata = SimpleNamespace(
        external_flashmla=None,
        pcp_prolog=None if decode_only else mla_v1.PCPMLAPrologMetadata(torch.arange(3), torch.arange(6), 0),
        num_actual_tokens=0 if case == "padding_only_rank" else 3,
        num_decodes=0 if case == "padding_only_rank" else 1,
        num_prefills=0 if decode_only or case == "padding_only_rank" else 1,
        num_decode_tokens=0 if case == "padding_only_rank" else (3 if decode_only else 1),
    )
    hidden = torch.zeros(3, 128, dtype=torch.bfloat16)
    output = torch.empty(3, impl.num_heads * impl.v_head_dim, dtype=hidden.dtype)
    impl.mla_preprocess_prolog_v3 = MagicMock(return_value=(None, None))
    impl._mla_preprocess = MagicMock(return_value=(None, None, None))
    impl.o_proj = MagicMock(side_effect=lambda tensor, **_kwargs: (tensor,))
    with (
        patch.object(mla_v1, "_EXTRA_CTX", SimpleNamespace(num_tokens=3)),
        patch.object(mla_v1, "maybe_save_kv_layer_to_connector"),
    ):
        impl.forward("layer", hidden, history, metadata, output)

    assert impl.mla_preprocess_prolog_v3.call_count == int(expect_prolog)
    assert impl._mla_preprocess.call_count == int(not expect_prolog)


@pytest.mark.parametrize(
    "impl_type,num_actual,num_decode,split_history",
    [
        (AscendMLAImpl, 4, 2, True),
        (AscendMLAImpl, 4, 2, False),
        (AscendMlaDCPImpl, 4, 2, False),
        (AscendMLAImpl, 4, 4, False),  # All local queries are sharded decode.
        (AscendMLAImpl, 4, 0, False),
        (AscendMlaDCPImpl, 4, 0, False),
        (AscendMLAImpl, 0, 0, False),
        (AscendMlaDCPImpl, 0, 0, False),
    ],
)
def test_pcp_prolog_publishes_owned_kv_and_preserves_local_current(impl_type, num_actual, num_decode, split_history):
    """Cache ownership cannot erase current KV or skip an empty rank's collective."""
    impl = _make_impl(impl_type)
    impl.use_pcp = True
    impl.use_dcp = impl_type is AscendMlaDCPImpl
    impl.is_pcp_decode_sharded = not impl.use_dcp
    impl.speculative_config = None
    tokens = 5
    replicated = num_decode if impl.use_dcp else 0
    impl._pcp_prolog_v3_buffers = (
        torch.zeros(1, BLOCK_SIZE, 1, NOPE_DIM),
        torch.zeros(1, BLOCK_SIZE, 1, ROPE_DIM),
        torch.arange(tokens),
    )
    impl.fused_qkv_a_proj = MagicMock(
        side_effect=lambda value: (torch.zeros(value.shape[0], impl.q_lora_rank + FUSED_DIM),)
    )
    impl.q_a_layernorm = MagicMock(side_effect=lambda value: value)
    impl.q_proj = MagicMock(
        side_effect=lambda value: (torch.full((value.shape[0], impl.num_heads * impl.qk_head_dim), 91.0),)
    )
    impl.rope_single = MagicMock(side_effect=lambda value, _cos, _sin: value)
    impl.kv_b_proj = MagicMock(
        side_effect=lambda value: (
            torch.zeros(value.shape[0], impl.num_heads * (impl.qk_nope_head_dim + impl.v_head_dim)),
        )
    )
    local_slots = torch.tensor([8, -1, 12, -1, -1]) if replicated else torch.tensor([12, -1, 14, 15, -1])
    if not num_actual:
        local_slots.fill_(-1)
    remote_slots = torch.tensor([8, 9, 10, 11, -1])
    if split_history:
        # Exercise non-monotonic slots on both sides of a history page boundary.
        remote_slots[:2] = torch.tensor([BLOCK_SIZE + 1, BLOCK_SIZE - 1])
        local_slots[0] = BLOCK_SIZE
    # Only rank 0 carries replicated-prefix slots in the runner's physical mapping.
    remote_slots[:replicated] = local_slots[:replicated]
    rank_one_slots = local_slots.clone()
    rank_one_slots[:replicated] = -1
    hidden = torch.arange(tokens, dtype=torch.float32)[:, None].expand(-1, 128).contiguous()
    positions = torch.arange(tokens, dtype=torch.float32)[:, None].expand(-1, ROPE_DIM).contiguous()
    metadata = SimpleNamespace(
        pcp_prolog=None,
        uses_pcp_sharded_layout=True,
        pcp_cos=positions,
        pcp_sin=-positions,
        num_actual_tokens=num_actual,
        num_decode_tokens=num_decode,
        num_prefills=int(num_actual > num_decode),
        external_flashmla=None,
        causal=True,
        slot_mapping=torch.cat((remote_slots, rank_one_slots)),
        decode=SimpleNamespace(actual_seq_lengths_q=[num_decode]) if num_decode else None,
        prefill=SimpleNamespace(
            cos=torch.ones(tokens - num_decode, ROPE_DIM),
            sin=torch.zeros(tokens - num_decode, ROPE_DIM),
            pcp_local_num_input_tokens=tokens,
        )
        if num_actual > num_decode
        else None,
    )
    history = (
        (
            torch.empty(2, BLOCK_SIZE, 1, NOPE_DIM, dtype=torch.bfloat16),
            torch.empty(2, BLOCK_SIZE, 1, ROPE_DIM, dtype=torch.bfloat16),
        )
        if split_history
        else _component_major_cache()
    )
    for tensor in history:
        tensor.zero_()
    events = []
    local_nope = torch.arange(tokens, dtype=torch.float32) + 20
    local_rope = local_nope + 30
    remote_nope = torch.arange(tokens - replicated, dtype=torch.float32) + 100
    remote_rope = remote_nope + 30

    def prolog(**kwargs):
        events.append("prolog")
        torch.testing.assert_close(kwargs["cache_index"], torch.arange(tokens))
        torch.testing.assert_close(kwargs["token_x"], hidden)
        torch.testing.assert_close(kwargs["rope_cos"], positions)
        torch.testing.assert_close(kwargs["rope_sin"], -positions)
        assert kwargs["query_quant_mode"] == 0
        kwargs["kv_cache"].flatten(0, 1)[:tokens].copy_(local_nope[:, None, None].expand(-1, 1, NOPE_DIM))
        kwargs["kr_cache"].flatten(0, 1)[:tokens].copy_(local_rope[:, None, None].expand(-1, 1, ROPE_DIM))
        q_rows = torch.arange(tokens, dtype=torch.float32) + 200
        return (
            q_rows[:, None].expand(-1, impl.num_heads * NOPE_DIM).contiguous(),
            q_rows[:, None].expand(-1, impl.num_heads * ROPE_DIM).contiguous(),
            None,
            None,
            None,
        )

    def gather(packed, dim):
        events.append("pcp_gather")
        assert dim == 0 and packed.shape == (tokens - replicated, 1, FUSED_DIM)
        torch.testing.assert_close(packed[:, 0, 0], local_nope[replicated:])
        remote = torch.cat(
            (remote_nope[:, None, None].expand(-1, 1, NOPE_DIM), remote_rope[:, None, None].expand(-1, 1, ROPE_DIM)),
            dim=-1,
        )
        return torch.cat((remote, packed), dim=0)

    def scatter(*, key, value, key_cache, value_cache, slot_mapping):
        events.append("scatter")
        for row, slot in enumerate(slot_mapping.tolist()):
            if slot >= 0:
                key_cache[slot // BLOCK_SIZE, slot % BLOCK_SIZE] = key[row]
                value_cache[slot // BLOCK_SIZE, slot % BLOCK_SIZE] = value[row]

    def gather_q(*fragments, dim):
        events.append("dcp_q")
        assert dim == 1
        return fragments

    impl._dcp_all_gather_fragments = MagicMock(side_effect=gather_q)
    group = SimpleNamespace(world_size=2, rank_in_group=1, all_gather=MagicMock(side_effect=gather))
    builder = mla_v1.AscendMLAMetadataBuilder.__new__(mla_v1.AscendMLAMetadataBuilder)
    builder.use_mla_rope = True
    builder.rope_dim = ROPE_DIM
    builder.is_pcp_decode_sharded = impl.is_pcp_decode_sharded
    builder._pcp_prolog_layers = (
        SimpleNamespace(impl=SimpleNamespace(_uses_pcp_prolog_v3=lambda _cache: True), kv_cache=history),
    )
    global_prefill = num_actual == 0 or num_actual > num_decode
    common = SimpleNamespace(
        pcp_context=SimpleNamespace(global_batch=SimpleNamespace(is_prefilling_np=torch.tensor([global_prefill])))
    )
    with (
        patch.object(mla_v1, "get_pcp_group", return_value=group),
        patch(
            "vllm_ascend.attention.context_parallel.mla_cp._EXTRA_CTX",
            SimpleNamespace(is_draft_model=False, is_draft_model_prefill=False),
        ),
        patch("torch_npu.npu_mla_prolog_v3", side_effect=prolog) as operator,
        patch("torch_npu.npu_kv_rmsnorm_rope_cache") as legacy,
        patch.object(mla_v1.DeviceOperator, "reshape_and_cache", side_effect=scatter),
        patch.object(mla_v1, "wait_for_kv_layer_from_connector", side_effect=lambda _name: events.append("wait")),
        patch.object(mla_v1, "notify_kv_cache_written", side_effect=lambda _name: events.append("notify")),
    ):
        builder._build_pcp_prolog_metadata(metadata, common)
        assert metadata.pcp_prolog is not None
        decode, prefill = impl.mla_preprocess_prolog_v3(hidden, history, metadata)

    operator.assert_called_once()
    legacy.assert_not_called()
    expected_events = ["wait", "prolog"] + ["scatter"] * bool(replicated) + ["pcp_gather", "scatter"]
    if num_decode and impl.use_dcp:
        expected_events.append("dcp_q")
    assert events == expected_events + ["notify"]
    group.all_gather.assert_called_once()
    expected_nope = dict(zip(remote_slots[replicated:].tolist(), remote_nope.tolist()))
    expected_nope.update(zip(local_slots.tolist(), local_nope.tolist()))
    for slot, value in expected_nope.items():
        if slot >= 0:
            assert history[0][slot // BLOCK_SIZE, slot % BLOCK_SIZE, 0, 0] == value
            assert history[1][slot // BLOCK_SIZE, slot % BLOCK_SIZE, 0, 0] == value + 30
    if split_history:
        # Padding must leave *all* unowned rows untouched on both pages.
        owned = {slot for slot in expected_nope if slot >= 0}
        for component in history:
            rows = component.flatten(0, 1)
            unowned = [slot for slot in range(len(rows)) if slot not in owned]
            assert torch.count_nonzero(rows[unowned]) == 0
    assert (decode is not None) == bool(num_decode)
    assert (prefill is not None) == bool(num_actual > num_decode)
    if num_decode:
        torch.testing.assert_close(decode.ql_nope[:, 0, 0], torch.arange(num_decode, dtype=torch.float32) + 200)
        if impl.use_dcp:
            assert events.index("pcp_gather") < events.index("dcp_q")
            torch.testing.assert_close(decode.current_k_nope[:, 0, 0], local_nope[:num_decode])
            torch.testing.assert_close(decode.current_k_pe[:, 0, 0], local_rope[:num_decode])
    if prefill is not None:
        prefill_rows = num_actual - num_decode
        torch.testing.assert_close(
            impl.kv_b_proj.call_args.args[0].reshape(prefill_rows, -1)[:, 0], local_nope[num_decode:num_actual]
        )
        torch.testing.assert_close(prefill.k_pe[:, 0, 0], local_rope[num_decode:num_actual])
        torch.testing.assert_close(prefill.q_nope[:, 0, 0], torch.full((prefill_rows,), 91.0))


@pytest.mark.parametrize(
    "width,hidden_rows,rope_rows,actual,decode,capacity",
    [(0, 0, 0, 0, 0, 0), (2, 1, 2, 1, 0, 2), (2, 2, 1, 1, 0, 2), (2, 2, 2, 1, 2, 2), (2, 2, 2, 1, 0, 1)],
    ids=["empty-global-batch", "short-hidden", "short-rope", "decode-exceeds-actual", "short-workspace"],
)
def test_pcp_prolog_checks_padded_input_before_write(width, hidden_rows, rope_rows, actual, decode, capacity):
    impl = _make_impl()
    impl.use_pcp = True
    impl._pcp_prolog_v3_buffers = (
        torch.empty(1, BLOCK_SIZE, 1, NOPE_DIM),
        torch.empty(1, BLOCK_SIZE, 1, ROPE_DIM),
        torch.arange(capacity),
    )
    metadata = SimpleNamespace(
        pcp_prolog=mla_v1.PCPMLAPrologMetadata(torch.arange(width), torch.arange(2 * width), 0),
        pcp_cos=torch.zeros(rope_rows, ROPE_DIM),
        pcp_sin=torch.zeros(rope_rows, ROPE_DIM),
        num_actual_tokens=actual,
        num_decode_tokens=decode,
        external_flashmla=None,
        decode=None,
        prefill=None,
    )
    hidden = torch.zeros(hidden_rows, 128)
    history = _component_major_cache()
    with (
        patch.object(mla_v1, "get_pcp_group") as group,
        patch("torch_npu.npu_mla_prolog_v3") as prolog,
        patch.object(mla_v1.DeviceOperator, "reshape_and_cache") as scatter,
        patch.object(mla_v1, "notify_kv_cache_written") as notify,
    ):
        if width:
            reason = "workspace" if capacity < width else "padded inputs"
            with pytest.raises(RuntimeError, match=reason):
                impl.mla_preprocess_prolog_v3(hidden, history, metadata)
        else:
            assert impl.mla_preprocess_prolog_v3(hidden, history, metadata) == (None, None)
    prolog.assert_not_called()
    group.assert_not_called()
    scatter.assert_not_called()
    notify.assert_not_called()
