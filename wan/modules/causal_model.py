# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    rope_apply_dynamic,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    rope_params_angle,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
import torch.distributed as dist

from utils.debug_option import DEBUG

import torch.nn.functional as F

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
# flex_attention = torch.compile(
    # flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

# ----------------------------------------------
flex_attention = torch.compile(
    # flex_attention, dynamic=True, mode="max-autotune-no-cudagraphs")
    flex_attention, dynamic=True)


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)

def causal_rope_apply_dynamic(x, grid_sizes, freqs, shot_flags_for_rope, theta=1/6, start_frame=0):
    # print(f"[DEBUG] start_frame is {start_frame}, shot_flags is {shot_flags_for_rope}")
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            (freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1) + shot_flags_for_rope.view(f, 1, 1, -1) * theta).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        freqs_i = torch.polar(torch.ones_like(freqs_i), freqs_i)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)

class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        # When keys are stored without RoPE (use_wo_rope_cache=True), re-apply
        # RoPE to pinned sink tokens at a compact/fixed temporal position.  This
        # mirrors the LongLive/Infinite-RoPE style sink behavior where the sink
        # remains in-range instead of drifting to a very large historical index.
        self.sink_rope_start_frame = 0
        self.qk_norm = qk_norm
        self.eps = eps
        # Support list/tuple local_attn_size by converting to list first (handles OmegaConf ListConfig)
        if not isinstance(local_attn_size, int) and hasattr(local_attn_size, "__iter__"):
            values = list(local_attn_size)
        else:
            values = [int(local_attn_size)]
        non_neg_vals = [int(v) for v in values if int(v) != -1]
        max_local = max(non_neg_vals) if len(non_neg_vals) > 0 else -1
        self.max_attention_size = 32760 if max_local == -1 else max_local * 1560
        # print(f"Self.max_attention_size is {self.max_attention_size}")
        
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        ###
        kv_cache_context=None,
        shot_flags_for_rope=None,
        freqs_dynamic=None,
        use_wo_rope_cache=False,
        ###
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None and kv_cache_context is None:  # for causal training, no kv cache
            
            if shot_flags_for_rope is None:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)
            else:
                roped_query = rope_apply_dynamic(q, grid_sizes, freqs=freqs_dynamic, shot_flags_for_rope=shot_flags_for_rope).type_as(v)
                roped_key = rope_apply_dynamic(k, grid_sizes, freqs=freqs_dynamic, shot_flags_for_rope=shot_flags_for_rope).type_as(v)

            x = F.scaled_dot_product_attention(
                roped_query.transpose(2, 1), 
                roped_key.transpose(2, 1), 
                v.transpose(2, 1), 
                attn_mask=block_mask, dropout_p=0.0, is_causal=False
            ).transpose(2, 1)
        
        # [TODO]
        elif kv_cache is not None and kv_cache_context is not None:  # for kv cache, with context cache

            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            # Add for Changing Rope Start Frames because of Condition Latents
            condition_start_frame = kv_cache_context['k'].shape[1] // frame_seqlen
            # print(f"Condition start frame is {condition_start_frame}")
            # print(f"Rope Start frame is {current_start_frame + condition_start_frame}")
            if shot_flags_for_rope is None:
                roped_query = causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=current_start_frame + condition_start_frame).type_as(v)  # [b num_token head channel]
                roped_key = causal_rope_apply(
                    k, grid_sizes, freqs, start_frame=current_start_frame + condition_start_frame).type_as(v)
            else:
                if use_wo_rope_cache:
                    rope_start_frame = min(current_start_frame + condition_start_frame, condition_start_frame + 18)
                else:
                    rope_start_frame = current_start_frame + condition_start_frame
                #     rope_start_frame = current_start_frame + condition_start_frame
                #     rope_end_frame = rope_start_frame + q.shape[1] // 1560
                #     # q shape is [b, n, head, c]
                #     roped_query = causal_rope_apply_dynamic(
                #     q, grid_sizes, freqs=freqs_dynamic, shot_flags_for_rope=shot_flags_for_rope, start_frame=rope_start_frame).type_as(v)  # [b num_token head channel]
                #     print("【Debug】Query")
                #     print(f"【Debug】Rope_start_frame is {rope_start_frame}")
                #     print(f"【Debug】shot_flags_for_rope is {shot_flags_for_rope}")
                # else:
                roped_query = causal_rope_apply_dynamic(
                    q, grid_sizes, freqs=freqs_dynamic, shot_flags_for_rope=shot_flags_for_rope, start_frame=rope_start_frame).type_as(v)  # [b num_token head channel]
                # print(f"[DEBUG] w rope cache rope_start_index={current_start_frame + condition_start_frame}, shot_flags_for_rope={shot_flags_for_rope}")
                roped_key = causal_rope_apply_dynamic(
                    k, grid_sizes, freqs=freqs_dynamic, shot_flags_for_rope=shot_flags_for_rope, start_frame=current_start_frame + condition_start_frame).type_as(v)

            num_new_tokens = roped_query.shape[1]
            query_frame_count = max(1, num_new_tokens // frame_seqlen)
            
            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen  # sink_size=3
            cached_local_end_before_insert = kv_cache["local_end_index"].item()
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]

            # Compute cache update parameters without modifying kv_cache directly
            cache_update_info = None

            if current_start_frame == 21:
                pass
            is_recompute = current_end <= kv_cache["global_end_index"].item() and current_start > 0
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size  # The token need to be dropped in the kvcache
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens  # The token need to be rolled (translated)

                # Compute updated local indices
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens  # the end index of the kvcache
                local_start_index = local_end_index - num_new_tokens  # start index of the kvcache
                                                                                               
                # Construct full k, v for attention computation (without modifying the original cache)
                # Create temporary k, v for computation
                temp_k = kv_cache["k"].clone()
                temp_v = kv_cache["v"].clone()
                
                # Apply rolling update to the temporary cache
                temp_k[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    temp_k[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                temp_v[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    temp_v[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                
                # Insert new key/value into the temporary cache
                # Protect sink_tokens only during recomputation; regular forward generation allows writing into the initial sink region
                write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
                roped_offset = max(0, write_start_index - local_start_index)
                write_len = max(0, local_end_index - write_start_index)
                if write_len > 0:
                    if use_wo_rope_cache:
                        temp_k[:, write_start_index:local_end_index] = k[:, roped_offset:roped_offset + write_len]
                    else:
                        temp_k[:, write_start_index:local_end_index] = roped_key[:, roped_offset:roped_offset + write_len]
                    temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]

                # Save cache update info for later use
                cache_update_info = {
                    "action": "roll_and_insert",
                    "sink_tokens": sink_tokens,
                    "num_rolled_tokens": num_rolled_tokens,
                    "num_evicted_tokens": num_evicted_tokens,
                    "local_start_index": local_start_index,
                    "local_end_index": local_end_index,
                    "write_start_index": write_start_index,
                    "write_end_index": local_end_index,
                    "new_k": roped_key[:, roped_offset:roped_offset + write_len] if not use_wo_rope_cache else None,
                    "new_k_wo_rope": k[:, roped_offset:roped_offset + write_len],  # add for without rope
                    "new_v": v[:, roped_offset:roped_offset + write_len],
                    "current_end": current_end,
                    "is_recompute": is_recompute
                }
                print(f"cut the kv cache!")
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                
                # Construct full k, v for attention computation (without modifying the original cache)
                temp_k = kv_cache["k"].clone()
                temp_v = kv_cache["v"].clone()
                # Protect sink_tokens only during recomputation; regular forward generation allows writing into the initial sink region
                write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
                roped_offset = max(0, write_start_index - local_start_index)
                write_len = max(0, local_end_index - write_start_index)
                if write_len > 0:
                    if use_wo_rope_cache:
                        temp_k[:, write_start_index:local_end_index] = k[:, roped_offset:roped_offset + write_len]
                    else:
                        temp_k[:, write_start_index:local_end_index] = roped_key[:, roped_offset:roped_offset + write_len]
                    temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]
                # print(f"[DEBUG] Write: write_start_index={write_start_index}, local_end_index={local_end_index}, use roped_offset={roped_offset} roped_offset + write_len={roped_offset + write_len}")
                # Save cache update info for later use
                cache_update_info = {
                    "action": "direct_insert",
                    "local_start_index": local_start_index,
                    "local_end_index": local_end_index,
                    "write_start_index": write_start_index,
                    "write_end_index": local_end_index,
                    "new_k": roped_key[:, roped_offset:roped_offset + write_len] if not use_wo_rope_cache else None,
                    "new_k_wo_rope": k[:, roped_offset:roped_offset + write_len],  # add for without rope
                    "new_v": v[:, roped_offset:roped_offset + write_len],
                    "current_end": current_end,
                    "is_recompute": is_recompute
                }


            # Use temporary k, v to compute attention
            if sink_tokens > 0:  # [TODO]
                # Concatenate sink tokens and local window tokens, keeping total length strictly below max_attention_size
                # Only frames that were already present in the cache before this
                # forward call may act as sink. Newly generated frames should first
                # be generated with normal/local RoPE, then become sink for later
                # blocks after the clean-cache update.
                active_sink_tokens = min(sink_tokens, cached_local_end_before_insert, local_end_index)
                local_budget = self.max_attention_size - active_sink_tokens
                k_sink = temp_k[:, :active_sink_tokens]
                if use_wo_rope_cache:
                    sink_frame_count = k_sink.shape[1] // frame_seqlen
                    if sink_frame_count > 0:
                        grid_sizes_for_sink = grid_sizes.clone()
                        grid_sizes_for_sink[0][0] = sink_frame_count
                        sink_flag = 0
                        if shot_flags_for_rope is not None and shot_flags_for_rope.numel() > 0:
                            sink_flag = int(shot_flags_for_rope[0].item())
                        shot_flags_for_rope_sink = torch.full(
                            (sink_frame_count,),
                            sink_flag,
                            dtype=torch.int32,
                            device=v.device,
                        )
                        k_sink = causal_rope_apply_dynamic(
                            k_sink,
                            grid_sizes_for_sink,
                            freqs=freqs_dynamic,
                            shot_flags_for_rope=shot_flags_for_rope_sink,
                            start_frame=condition_start_frame + int(self.sink_rope_start_frame),
                        ).type_as(v)
                v_sink = temp_v[:, :active_sink_tokens]
                # add for context
                k_context =  kv_cache_context["k"].clone()
                v_context = kv_cache_context["v"].clone()

                if local_budget > 0:
                    local_start_for_window = max(active_sink_tokens, local_end_index - local_budget)
                    k_local = temp_k[:, local_start_for_window:local_end_index]
                    local_frame_count = k_local.shape[1] // frame_seqlen
                    if use_wo_rope_cache and local_frame_count > 0:
                        local_key_rope_start_frame = max(
                            condition_start_frame + (active_sink_tokens // frame_seqlen),
                            rope_start_frame - max(0, local_frame_count - query_frame_count),
                        )
                        shot_flags_for_rope_local = torch.full(
                            (local_frame_count,),
                            int(shot_flags_for_rope[0].item()),
                            dtype=torch.int32,
                            device=v.device,
                        )
                        grid_sizes_for_local = grid_sizes.clone()
                        grid_sizes_for_local[0][0] = local_frame_count
                        k_local = causal_rope_apply_dynamic(k_local, grid_sizes_for_local, freqs=freqs_dynamic, shot_flags_for_rope=shot_flags_for_rope_local, start_frame=local_key_rope_start_frame).type_as(v)
                    # print(f"use wo rope cache")
                    # print(f"rope_start_frame is {rope_start_frame}")
                    # print(f"shot_flags_for_rope is {shot_flags_for_rope}")
                    v_local = temp_v[:, local_start_for_window:local_end_index]
                    k_cat = torch.cat([k_sink, k_local, k_context], dim=1)
                    v_cat = torch.cat([v_sink, v_local, v_context], dim=1)
                    # print(f"[DEBUG] Local local_start_for_window={local_start_for_window}, local_end_index={local_end_index}")
                else:
                    k_cat = torch.cat([k_sink, k_context], dim=1)
                    v_cat = torch.cat([v_sink, v_context], dim=1)
                x = attention(
                    roped_query,
                    k_cat,
                    v_cat
                )
            else:
                window_start = max(0, local_end_index - self.max_attention_size)
                k_context =  kv_cache_context["k"].clone()
                v_context = kv_cache_context["v"].clone()
                k_local = temp_k[:, window_start:local_end_index]
                v_local = temp_v[:, window_start:local_end_index]
                local_frame_count = k_local.shape[1] // frame_seqlen
                if use_wo_rope_cache and local_frame_count > 0:
                    local_key_rope_start_frame = max(
                        condition_start_frame,
                        rope_start_frame - max(0, local_frame_count - query_frame_count),
                    )
                    shot_flags_for_rope_local = torch.full(
                        (local_frame_count,),
                        int(shot_flags_for_rope[0].item()),
                        dtype=torch.int32,
                        device=v.device,
                    )
                    grid_sizes_for_local = grid_sizes.clone()
                    grid_sizes_for_local[0][0] = local_frame_count
                    k_local = causal_rope_apply_dynamic(k_local, grid_sizes_for_local, freqs=freqs_dynamic, shot_flags_for_rope=shot_flags_for_rope_local, start_frame=local_key_rope_start_frame).type_as(v)
                    # print(f"[DEBUG] Key: rope_start_frame is {rope_start_frame}  Key latent number is {k_local.shape[1] // 1560}  shot_flags_for_rope is {shot_flags_for_rope_local}")

                k_cat = torch.cat([k_context, k_local], dim=1)
                v_cat = torch.cat([v_context, v_local], dim=1)
                x = attention(
                    roped_query,
                    k_cat,
                    v_cat
                )

        elif kv_cache is None and kv_cache_context is not None:  # Update the context kv cache
            current_end, local_end_index = None, None
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            if shot_flags_for_rope is None:
                roped_query = causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=0).type_as(v)  # [b num_token head channel]
                roped_key = causal_rope_apply(
                    k, grid_sizes, freqs, start_frame=0).type_as(v)
            else:
                roped_query = causal_rope_apply_dynamic(
                    q, grid_sizes, freqs=freqs_dynamic, shot_flags_for_rope=shot_flags_for_rope, start_frame=0).type_as(v)  # [b num_token head channel]
                roped_key = causal_rope_apply_dynamic(
                    k, grid_sizes, freqs=freqs_dynamic, shot_flags_for_rope=shot_flags_for_rope, start_frame=0).type_as(v)

            cache_update_info = {
                "action": "context_insert",
                "context_new_k": roped_key,  # [b, n, head, c]
                "context_new_v": v,
                }
            
            x = attention(
                roped_query,
                roped_key,
                v
            )

        # output
        x = x.flatten(2)
        x = self.o(x)
        
        # Return both output and cache update info
        if kv_cache is not None or kv_cache_context is not None:
            return x, (current_end, local_end_index, cache_update_info)
        else:
            return x
        

class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        ###
        kv_cache_context=None,
        shot_flags_for_rope=None,
        text_context_indices=None,
        freqs_dynamic=None,
        use_wo_rope_cache=False,
        ###
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        if shot_flags_for_rope is not None:
            self_attn_result = self.self_attn(
                (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
                seq_lens, grid_sizes,
                freqs, block_mask, kv_cache, current_start, cache_start, kv_cache_context=kv_cache_context, freqs_dynamic=freqs_dynamic, shot_flags_for_rope=shot_flags_for_rope, use_wo_rope_cache=use_wo_rope_cache)
        else:
            self_attn_result = self.self_attn(
                (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
                seq_lens, grid_sizes,
                freqs, block_mask, kv_cache, current_start, cache_start, kv_cache_context=kv_cache_context, use_wo_rope_cache=use_wo_rope_cache)
        
        if kv_cache is not None or kv_cache_context is not None:
            y, cache_update_info = self_attn_result
        else:
            y = self_attn_result
            cache_update_info = None

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None, grid_sizes=None, shot_flags_for_rope=None, text_context_indices=None):
            x = x + self.cross_attn(
                self.norm3(x),
                context,
                context_lens,
                crossattn_cache=crossattn_cache,
                grid_sizes=grid_sizes,
                shot_flags_for_rope=shot_flags_for_rope,
                text_context_indices=text_context_indices,
            )
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache, grid_sizes, shot_flags_for_rope, text_context_indices)
        
        if cache_update_info is not None:
            # cache_update_info is already in the format (current_end, local_end_index, cache_update_info)
            return x, cache_update_info
        else:
            return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.d = d  # for dynamic rope

        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    def _apply_cache_updates(self, kv_cache, cache_update_infos, use_wo_rope_cache=False):
        """
        Applies cache updates collected from multiple blocks.
        Args:
            kv_cache: List of cache dictionaries for each block
            cache_update_infos: List of (block_index, cache_update_info) tuples
        """
        for block_index, (current_end, local_end_index, update_info) in cache_update_infos:
            if update_info is not None:
                cache = kv_cache[block_index]
                
                if update_info["action"] == "roll_and_insert":
                    # Apply rolling update
                    sink_tokens = update_info["sink_tokens"]
                    num_rolled_tokens = update_info["num_rolled_tokens"]
                    num_evicted_tokens = update_info["num_evicted_tokens"]
                    local_start_index = update_info["local_start_index"]
                    local_end_index = update_info["local_end_index"]
                    write_start_index = update_info.get("write_start_index", local_start_index)
                    write_end_index = update_info.get("write_end_index", local_end_index)
                    if use_wo_rope_cache:
                        new_k = update_info["new_k_wo_rope"]
                    else:
                        new_k = update_info["new_k"]
                    new_v = update_info["new_v"]
                    
                    # Perform the rolling operation
                    cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    
                    # Insert new key/value
                    if write_end_index > write_start_index and new_k.shape[1] == (write_end_index - write_start_index):
                        cache["k"][:, write_start_index:write_end_index] = new_k
                        cache["v"][:, write_start_index:write_end_index] = new_v
                    
                elif update_info["action"] == "direct_insert":
                    # Direct insert
                    local_start_index = update_info["local_start_index"]
                    local_end_index = update_info["local_end_index"]
                    write_start_index = update_info.get("write_start_index", local_start_index)
                    write_end_index = update_info.get("write_end_index", local_end_index)
                    if use_wo_rope_cache:
                        new_k = update_info["new_k_wo_rope"]
                    else:
                        new_k = update_info["new_k"]
                    new_v = update_info["new_v"]
                    
                    # Insert new key/value
                    if write_end_index > write_start_index and new_k.shape[1] == (write_end_index - write_start_index):
                        cache["k"][:, write_start_index:write_end_index] = new_k
                        cache["v"][:, write_start_index:write_end_index] = new_v
            
            # Update indices: do not roll back pointers during recomputation
            is_recompute = False if update_info is None else update_info.get("is_recompute", False)
            if not is_recompute:
                kv_cache[block_index]["global_end_index"].fill_(current_end)
                kv_cache[block_index]["local_end_index"].fill_(local_end_index)

    def _apply_cache_updates_context(self, kv_cache_context, cache_update_infos):
        """
        Applies cache updates collected from multiple blocks.
        Args:
            kv_cache: List of cache dictionaries for each block
            cache_update_infos: List of (block_index, cache_update_info) tuples
        """
        for block_index, (current_end, local_end_index, update_info) in cache_update_infos:
            if update_info is not None:
                # cache = kv_cache_context[block_index]
                if update_info["action"] == "context_insert":
                    kv_cache_context[block_index]['k'][:] = update_info['context_new_k']
                    kv_cache_context[block_index]['v'][:] = update_info['context_new_v']

    def _forward_inference(
        self,
        x,  # [b c f h w], f is frame number of one chunk, set as 3 in this case, torch.Size([1, 16, 3, 60, 104])
        t,  # [b f]
        context,  # [b n c] [1 512, 4096]
        seq_len,  # 327600 = 1560 * 21
        clip_fea=None,
        y=None,
        kv_cache: dict = None,  # list, 30, kv_cache[0].keys() dict_keys(['k', 'v', 'global_end_index', 'local_end_index'])
        crossattn_cache: dict = None,  # list, 30, crossattn_cache[0].keys() dict_keys(['k', 'v', 'is_init'])
        ###
        kv_cache_context: dict = None,  # list, 30, kv_cache_context[0].keys() dict_keys(['k', 'v', 'is_init'])
        shot_flags_for_rope: torch.Tensor = None,
        text_context_indices: torch.Tensor = None,
        use_wo_rope_cache: bool = False,  # whether use rope cache
        ###
        current_start: int = 0,  # token number
        cache_start: int = 0
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        d = self.d
        if shot_flags_for_rope is not None:  # use dynamic rope
            self.freqs_dynamic = torch.cat([
                rope_params_angle(1024, d - 4 * (d // 6)),
                rope_params_angle(1024, 2 * (d // 6)),
                rope_params_angle(1024, 2 * (d // 6))
            ],
                dim=1)
            if self.freqs_dynamic.device != device:
                self.freqs_dynamic = self.freqs_dynamic.to(device)
        else:
            self.freqs_dynamic = None

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        
        # print(f"x.device: {x[0].device}, t.device: {t.device}, context.device: {context.device}, seq_len: {seq_len}")

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        # print("patch embedding done")
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]  # list [b, n, c]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)  # [b n c] torch.Size([1, 4680, 1536])
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32
        # print("time embedding done")
        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))
        # print("text embedding done")
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            ###
            freqs_dynamic=self.freqs_dynamic,
            shot_flags_for_rope=shot_flags_for_rope,
            text_context_indices=text_context_indices,
            use_wo_rope_cache=use_wo_rope_cache,
            ###
        )
        # print("kwargs done")
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        cache_update_info = None
        cache_update_infos = []  # Collect cache update info for all blocks
        for block_index, block in enumerate(self.blocks):
            # print(f"block_index: {block_index}")
            # Activation checkpointing re-runs the block during backward.  In
            # causal rollout/training the KV caches are intentionally mutated
            # after the original forward (see _apply_cache_updates below), so
            # recomputation would observe a newer cache state than the original
            # checkpointed forward.  That can change sink/local-window branches
            # and trip PyTorch's non-reentrant checkpoint metadata checks
            # ("different number of tensors saved").  Keep checkpointing for
            # stateless/non-cache forwards, but run cacheful forwards normally.
            can_checkpoint = (
                torch.is_grad_enabled()
                and self.gradient_checkpointing
                and kv_cache is None
                and kv_cache_context is None
            )
            if can_checkpoint:
                kwargs.update(
                    {
                        # "kv_cache": kv_cache[block_index],
                        "kv_cache": kv_cache[block_index] if kv_cache else None,
                        "current_start": current_start,
                        "cache_start": cache_start,
                        ###
                        "kv_cache_context": kv_cache_context[block_index] if kv_cache_context else None,
                        ###
                    }
                )
                # print(f"forward checkpointing")
                result = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
                # Handle the result
                if (kv_cache is not None or kv_cache_context is not None) and isinstance(result, tuple):
                    x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                    # Extract base info for subsequent blocks (without concrete cache update details)
                    cache_update_info = block_cache_update_info[:2]  # (current_end, local_end_index)
                else:
                    x = result
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index] if kv_cache else None,
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        ###
                        "kv_cache_context": kv_cache_context[block_index] if kv_cache_context else None,
                        ###
                    }
                )
                # print(f"forward no checkpointing")
                result = block(x, **kwargs)
                # Handle the result
                if (kv_cache is not None or kv_cache_context is not None) and isinstance(result, tuple):
                    x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                    # Extract base info for subsequent blocks (without concrete cache update details)
                    cache_update_info = block_cache_update_info[:2]  # (current_end, local_end_index)
                else:
                    x = result
        # log_gpu_memory(f"in _forward_inference: {x[0].device}")
        # After all blocks are processed, apply cache updates in a single pass
        if kv_cache is not None and cache_update_infos:
            self._apply_cache_updates(kv_cache, cache_update_infos, use_wo_rope_cache=use_wo_rope_cache)
        if kv_cache is None and kv_cache_context is not None and cache_update_infos:  # only update context cache once
            self._apply_cache_updates_context(kv_cache_context, cache_update_infos)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
        ## add for condition input
        causal_use_condition_mask=False,
        condition_frame_number=None,
        shot_flags_for_rope=None,
        text_context_indices=None,
        local_attn_size=None,
        sink_size=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # pass
        # raise NotImplementedError()
    
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
        
        d = self.d
        if shot_flags_for_rope is not None:  # use dynamic rope
            self.freqs_dynamic = torch.cat([
                rope_params_angle(1024, d - 4 * (d // 6)),
                rope_params_angle(1024, 2 * (d // 6)),
                rope_params_angle(1024, 2 * (d // 6))
            ],
                dim=1)
            if self.freqs_dynamic.device != device:
                self.freqs_dynamic = self.freqs_dynamic.to(device)
        else:
            self.freqs_dynamic = None


        frame_seqlen = x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2])
        frame_seqlen_all = x.shape[2] * frame_seqlen
        condition_seqlen = condition_frame_number * frame_seqlen
        if sink_size is not None:
            sink_seqlen = sink_size * frame_seqlen
        iter_num = (x.shape[2] - condition_frame_number) // self.num_frame_per_block
        mask = torch.zeros([1, frame_seqlen_all])
        for i in range(iter_num):
            start_index = condition_seqlen + (i * self.num_frame_per_block * frame_seqlen)
            end_index = start_index + (1) * self.num_frame_per_block * frame_seqlen
            mask[::, start_index: end_index] = i + 1
        mask = (mask.unsqueeze(1).unsqueeze(-1) >= mask.unsqueeze(1).unsqueeze(1))
        mask[..., :condition_seqlen] = True
        if sink_size is not None:
            mask[..., condition_seqlen:, condition_seqlen: condition_seqlen+sink_seqlen] = True  # Fix for condition cannot see generation result
            for i in range(iter_num -1):  # [NOTE] 这里给sink=3写死了
                write_start_index = (condition_seqlen + sink_seqlen) + (i * self.num_frame_per_block * frame_seqlen)
                write_end_index = write_start_index + (1) * self.num_frame_per_block * frame_seqlen
                if local_attn_size is not None:
                    if write_end_index - condition_seqlen > local_attn_size * frame_seqlen:
                        write_end_index_stop = max(condition_seqlen+sink_seqlen, write_end_index-(local_attn_size-sink_size)*frame_seqlen)
                        mask[::, ::, write_start_index: write_end_index, condition_seqlen + sink_seqlen: write_end_index_stop] = False

        mask = mask.to(x.device)

        # import cv2
        # import numpy as np
        # mask_save = mask[0][0] * 255
        # mask_save = mask_save.cpu().numpy()
        # cv2.imwrite("mask_v1.jpg", np.uint8(mask_save))

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=mask,
            ###
            freqs_dynamic=self.freqs_dynamic,
            shot_flags_for_rope=shot_flags_for_rope,
            text_context_indices=text_context_indices,
            ###
            )
            # block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)
        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)

        self.block_mask = None ## for dynamic mask
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None or kwargs.get('kv_cache_context', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
