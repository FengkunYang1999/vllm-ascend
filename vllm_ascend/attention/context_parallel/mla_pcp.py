# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PCP-only MLA preprocessing; shared attention stays in AscendMLAImpl."""

import torch
from vllm.distributed.parallel_state import get_pcp_group
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.offloader.base import NoopOffloader, get_offloader
from vllm.utils.math_utils import cdiv

from vllm_ascend.attention.mla_v1 import (
    AscendMLAImpl,
    AscendMLAMetadata,
    DecodeMLAPreprocessResult,
)
from vllm_ascend.attention.utils import notify_kv_cache_written, wait_for_kv_layer_from_connector
from vllm_ascend.device.device_op import DeviceOperator
from vllm_ascend.utils import enable_sfa


class AscendMlaPCPImpl(AscendMLAImpl):
    """Fuse local Q/KV preparation and publish current KV once per PCP step."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pcp_prolog_v3_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def supports_pcp_prolog_v3(self) -> bool:
        """Admit native BF16 PCP-only layers with resident projections."""
        return (
            self.vllm_config.use_v2_model_runner
            and self.use_pcp
            and not self.use_dcp
            and self.is_pcp_decode_sharded
            and not self.use_flashmla
            and not enable_sfa(self.vllm_config)
            and self.num_kv_heads == 1
            and isinstance(get_offloader(), NoopOffloader)
            and self.supports_prolog_v3_writer()
            and isinstance(self.fused_qkv_a_proj.quant_method, UnquantizedLinearMethod)
            and isinstance(self.q_proj.quant_method, UnquantizedLinearMethod)
        )

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        # The base class also refreshes existing private weights in place on reload.
        super().process_weights_after_loading(act_dtype)
        if self.supports_pcp_prolog_v3():
            self._prepare_prolog_v3_weights()
            self._prepare_pcp_prolog_v3_buffers()

    def _prepare_pcp_prolog_v3_buffers(self) -> None:
        """Allocate during model loading, before the runner sizes the KV cache."""
        if self._pcp_prolog_v3_buffers is not None:
            return
        capacity = max(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            max(self.vllm_config.compilation_config.cudagraph_capture_sizes or [0]),
        )
        block_size = self.vllm_config.cache_config.block_size
        shape = (cdiv(capacity, block_size), block_size, self.num_kv_heads)
        self._pcp_prolog_v3_buffers = (
            torch.empty((*shape, self.kv_lora_rank), dtype=self.dtype, device=self.W_UV.device),
            torch.empty((*shape, self.qk_rope_head_dim), dtype=self.dtype, device=self.W_UV.device),
            torch.arange(capacity, dtype=torch.int64, device=self.W_UV.device),
        )

    def _can_use_pcp_prolog_v3(self, kv_cache, attn_metadata: AscendMLAMetadata) -> bool:
        """Admission depends on layout, never on the rank's local request mix."""
        if not (
            self.supports_pcp_prolog_v3()
            and attn_metadata.uses_pcp_sharded_layout
            and attn_metadata.external_flashmla is None
            and self._pcp_prolog_v3_buffers is not None
            and getattr(self, "_prolog_v3_weights", None) is not None
            and isinstance(kv_cache, tuple)
            and len(kv_cache) == 2
        ):
            return False
        nope, rope = kv_cache
        return (
            nope.dim() == rope.dim() == 4
            and nope.is_contiguous()
            and rope.is_contiguous()
            and nope.dtype == rope.dtype == torch.bfloat16
            and nope.device == rope.device
            and nope.shape[:3] == rope.shape[:3]
            and nope.shape[2] == 1
            and nope.shape[-1] == self.kv_lora_rank
            and rope.shape[-1] == self.qk_rope_head_dim
        )

    def _mla_preprocess(self, layer_name, hidden_states, kv_cache, attn_metadata):
        if not self._can_use_pcp_prolog_v3(kv_cache, attn_metadata):
            return super()._mla_preprocess(layer_name, hidden_states, kv_cache, attn_metadata)
        decode, prefill = self.mla_preprocess_pcp_prolog_v3(layer_name, hidden_states, kv_cache, attn_metadata)
        # Fused QKV-A gates are excluded by supports_prolog_v3_writer(). The
        # shared forward retains standalone g_proj results when this is None.
        return decode, prefill, None

    def mla_preprocess_pcp_prolog_v3(self, layer_name, hidden_states, kv_cache, attn_metadata):
        """One exchange covers decode, prefill and ranks with only padding."""
        group = get_pcp_group()
        num_tokens = hidden_states.shape[0]
        actual = attn_metadata.num_actual_tokens
        decodes = attn_metadata.num_decode_tokens
        global_slots = attn_metadata.slot_mapping
        cos, sin = attn_metadata.pcp_cos, attn_metadata.pcp_sin
        if cos is None or sin is None:
            raise RuntimeError("PCP PROLOG_V3 requires padded RoPE metadata")
        if not (
            cos.shape[0] == sin.shape[0] == num_tokens
            and global_slots.numel() == group.world_size * num_tokens
            and 0 <= decodes <= actual <= num_tokens
        ):
            raise RuntimeError("PCP PROLOG_V3 input rows, slots and token counts must agree")
        if num_tokens == 0:
            return None, None
        buffers = self._pcp_prolog_v3_buffers
        if buffers is None or num_tokens > buffers[2].numel():
            raise RuntimeError("PCP PROLOG_V3 workspace does not cover the padded input")
        # The metadata builder already supplies slots in rank-major gather order.
        local_slots = global_slots.view(group.world_size, num_tokens)[group.rank_in_group]

        if actual > decodes or (
            self.layerwise_kv_cache_hook is not None and getattr(self.layerwise_kv_cache_hook, "layerwise", False)
        ):
            wait_for_kv_layer_from_connector(layer_name)
        if self.layerwise_kv_cache_hook is not None:
            self.layerwise_kv_cache_hook.wait_for_layer(layer_name)

        # Positive unique indices decouple the operator from PAD_SLOT_ID and
        # avoid reading new rows back out of the historical cache.
        nope_buffer, rope_buffer, indices = buffers
        num_blocks = cdiv(num_tokens, nope_buffer.shape[1])
        workspace = (nope_buffer[:num_blocks], rope_buffer[:num_blocks])
        prolog_outputs = self._run_prolog_v3(
            hidden_states, cos, sin, workspace, indices[:num_tokens], quantize_query=False
        )
        latent = workspace[0].flatten(0, 1)[:num_tokens]
        k_pe = workspace[1].flatten(0, 1)[:num_tokens]
        packed = torch.cat((latent.squeeze(1), k_pe.squeeze(1)), dim=-1)
        # Assignment clears NaN/Inf in padding too; multiplication would not.
        packed.masked_fill_((local_slots < 0).unsqueeze(-1), 0)
        gathered = group.all_gather(packed, dim=0)
        global_latent, global_rope = gathered.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        DeviceOperator.reshape_and_cache(
            key=global_latent.unsqueeze(1),
            value=global_rope.unsqueeze(1),
            key_cache=kv_cache[0],
            value_cache=kv_cache[1],
            slot_mapping=global_slots,
        )
        notify_kv_cache_written(layer_name)

        decode_result = prefill_result = None
        if decodes:
            heads = self._prolog_v3_weights["num_heads"]
            q_nope = prolog_outputs[0].view(num_tokens, heads, self.kv_lora_rank)[:decodes, : self.num_heads]
            q_pe = prolog_outputs[1].view(num_tokens, heads, self.qk_rope_head_dim)[:decodes, : self.num_heads]
            q_nope, q_pe = self.reorg_decode_q(q_nope, q_pe)
            decode_result = DecodeMLAPreprocessResult(q_nope, q_pe, kv_cache[0], kv_cache[1])
        if actual > decodes:
            # Prefill consumes ordinary Q, not the absorbed Q returned by V3.
            # Reuse the native adapter with already-written KV so it performs
            # neither another cache write nor another PCP collective.
            qkv_lora = self.fused_qkv_a_proj(hidden_states[:actual])[0]
            q_c, kv_no_split, _ = self._split_fused_qkv_lora(qkv_lora)
            q_c = self.q_a_layernorm(q_c)
            local_latent, local_rope = packed[:actual].split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
            prefill_result = self.mla_preprocess_prefill(
                q_c,
                kv_no_split,
                kv_cache,
                attn_metadata,
                written_kv=(local_rope.contiguous(), local_latent.contiguous()),
            )
        return decode_result, prefill_result
