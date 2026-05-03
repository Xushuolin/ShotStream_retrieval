# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.distributed as dist
from model.base import FrameConcatCausalModel
import math
import torch.nn.functional as F

from einops import rearrange

import peft
# from peft import get_peft_model_state_dict

class CausalInferenceArPipeline(FrameConcatCausalModel):
    def __init__(
            self,
            args,
            device,
    ):
        # Step 1: Initialize all models
        super().__init__(args, device)

        # hard code for Wan2.1-T2V-1.3B        
        self.num_transformer_blocks = getattr(self.generator.model, 'num_layers', 30)  # number of layers
        self.num_heads = getattr(self.generator.model, 'num_heads', 12)  # num of heads
        self.d = getattr(self.generator.model, 'd', 128)  # feature dim of each head

        self.frame_seq_length = 1560

        self.kv_cache1 = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.local_attn_size = getattr(args, "local_attn_size", 21)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.max_context_frames = getattr(args, "max_context_frames", 10)  # for dynamic sample frames
        self.dynamic_sample_frames = getattr(args, "dynamic_sample_frames", False)
        self.change_rope = getattr(args, "change_rope", False)
        self.semantic_retrieval_context = getattr(args, "semantic_retrieval_context", True)
        self.semantic_memory_frames_per_shot = getattr(args, "semantic_memory_frames_per_shot", 1)
        self.context_rope_max_id = getattr(args, "context_rope_max_id", 2)
        self.route_retrieval_by_prompt_similarity = getattr(args, "route_retrieval_by_prompt_similarity", False)
        self.route_retrieval_cosine_threshold = getattr(args, "route_retrieval_cosine_threshold", 0.80)
        self.restrict_max_length  = getattr(args, "restrict_max_length", 81)

        self.multi_caption  = getattr(args, "multi_caption", False)

        self.lora_config = None
        if hasattr(args, 'adapter') and args.adapter is not None:
            self.is_lora_enabled = True
            self.lora_config = args.adapter

    @torch.no_grad()
    def _encode_shot_query(self, shot_prompt: str, device: torch.device):
        cond = self.text_encoder(text_prompts=[shot_prompt])
        query = cond["prompt_embeds"][0].mean(dim=0).to(device)
        return query

    @torch.no_grad()
    def _should_use_retrieval(self, prev_query_embed: torch.Tensor, cur_query_embed: torch.Tensor):
        if prev_query_embed is None or cur_query_embed is None:
            return True, float("nan")
        sim = F.cosine_similarity(prev_query_embed, cur_query_embed, dim=0).item()
        # Two-way routing only:
        #   high similarity -> fast heuristic context sampling
        #   low similarity  -> semantic retrieval
        use_retrieval = sim < self.route_retrieval_cosine_threshold
        return use_retrieval, sim

    @torch.no_grad()
    def _build_candidate_entries(self, shot_flags: torch.Tensor, latent_gen_iter: int):
        candidate_entries = []
        unique_flags = torch.unique(shot_flags)
        for shot_flag in unique_flags.tolist():
            shot_flag = int(shot_flag)
            if shot_flag >= latent_gen_iter:
                continue
            indices = torch.where(shot_flags[0] == shot_flag)[0].tolist()
            if len(indices) == 0:
                continue
            picked = [indices[0]]
            if self.semantic_memory_frames_per_shot >= 2 and len(indices) > 1:
                picked.append(indices[1])
            for idx in picked:
                candidate_entries.append((idx, shot_flag))
        return candidate_entries

    def _to_local_context_flags(self, shot_flags_list):
        """
        Remap global shot ids to compact local ids for context-only references.
        """
        unique_sorted = sorted(set(int(x) for x in shot_flags_list))
        id_map = {sid: i for i, sid in enumerate(unique_sorted)}
        return [id_map[int(x)] for x in shot_flags_list]

    def _to_small_context_flags(self, shot_flags_list):
        """
        Map context shot ids to a small seen-like range for RoPE stability.
        """
        local_flags = self._to_local_context_flags(shot_flags_list)
        max_id = max(0, int(self.context_rope_max_id))
        return [min(int(v), max_id) for v in local_flags]

    @torch.no_grad()
    def _kv_retrieval_topk_history(
        self,
        history_video: torch.Tensor,
        candidate_entries,
        query_text: str,
        device: torch.device,
        dtype: torch.dtype,
    ):
        if len(candidate_entries) == 0:
            return None, None

        candidate_indices = [e[0] for e in candidate_entries]
        candidate_shot_flags = [e[1] for e in candidate_entries]
        candidate_frames = history_video[candidate_indices]
        candidate_frames = rearrange(candidate_frames, 'f h w c -> f c 1 h w').to(device).to(dtype)
        candidate_latents = self.vae.encode_to_latent(candidate_frames).to(device).to(dtype)
        candidate_latents = rearrange(candidate_latents, 'f 1 c h w -> 1 f c h w')

        batch_size = 1
        num_candidates = candidate_latents.shape[1]
        self._initialize_crossattn_cache(batch_size=batch_size, dtype=dtype, device=device)
        self._initialize_context_kv_cache(
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            kv_cache_size_override=num_candidates * self.frame_seq_length
        )

        conditional_dict = self.text_encoder(text_prompts=[query_text])
        retrieval_timestep = torch.zeros([batch_size, num_candidates], device=device, dtype=torch.int64)
        retrieval_shot_flags = torch.tensor(
            self._to_small_context_flags(candidate_shot_flags), dtype=torch.int32, device=device
        )
        self.generator(
            noisy_image_or_video=candidate_latents,
            conditional_dict=conditional_dict,
            timestep=retrieval_timestep,
            kv_cache=None,
            crossattn_cache=self.crossattn_cache,
            kv_cache_context=self.kv_cache1_context,
            shot_flags_for_rope=retrieval_shot_flags if self.change_rope else None,
            current_start=0,
            use_wo_rope_cache=False,
        )

        frame_token_count = self.frame_seq_length
        num_layers = len(self.kv_cache1_context)
        scores = []
        for cand_i in range(num_candidates):
            token_start = cand_i * frame_token_count
            token_end = token_start + frame_token_count
            layer_scores = []
            for layer_i in range(num_layers):
                q_layer = self.crossattn_cache[layer_i]["k"][0].mean(dim=0)  # [n, d]
                k_frame = self.kv_cache1_context[layer_i]["k"][0, token_start:token_end]  # [t, n, d]
                logits = (k_frame * q_layer.unsqueeze(0)).sum(dim=-1) / math.sqrt(q_layer.shape[-1])
                attn = torch.softmax(logits, dim=0)
                layer_scores.append((attn * logits).mean())
            scores.append(torch.stack(layer_scores).mean())

        scores = torch.stack(scores)
        top_k = min(self.max_context_frames, scores.shape[0])
        top_ids = torch.topk(scores, k=top_k, dim=0).indices.tolist()
        selected_indices = [candidate_indices[i] for i in top_ids]
        selected_shot_flags = [candidate_shot_flags[i] for i in top_ids]
        selected_shot_flags = self._to_small_context_flags(selected_shot_flags)
        if len(selected_indices) < self.max_context_frames and len(selected_indices) > 0:
            pad_num = self.max_context_frames - len(selected_indices)
            selected_indices += [selected_indices[-1]] * pad_num
            selected_shot_flags += [selected_shot_flags[-1]] * pad_num
        return selected_indices, selected_shot_flags
            
    @torch.no_grad()
    def inference(
        self,
        batch,
        use_wo_rope_cache=False,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        # Step 1: Prepare data
        device, dtype = self.vae.model.encoder.conv1.weight.device, self.vae.model.encoder.conv1.weight.dtype 

        # video_data_gt = torch.tensor(batch['data'][0]).to(device).to(dtype)  # [f h w c]
        global_captions = batch['global_captions']
        shots_captions = batch['shots_captions']
        shot_flags_gt = torch.tensor([batch['shot_flag']]).to(torch.int32) 
        shot_flags_unique_gt = torch.unique(shot_flags_gt)

        # Save generated results
        output_images_list = []
        shot_flags_output = []
        prev_shot_query_embed = None

        for latent_gen_iter in range(len(shot_flags_unique_gt)):
            print(f"shot {latent_gen_iter} begin")
            shot_flags = torch.tensor(shot_flags_output).to(torch.int32) 
            shot_flags_unique = torch.unique(shot_flags)
            current_shot_query = shots_captions[0][latent_gen_iter][0][0]
            current_shot_query_embed = self._encode_shot_query(current_shot_query, device)

            if self.dynamic_sample_frames:
                if latent_gen_iter > 0:
                    shot_number = shot_flags_unique.shape[0]  # except the noise latent 
                    base_count = self.max_context_frames // shot_number
                    remainder = self.max_context_frames % shot_number
                    counts = [base_count] * shot_number
                    for i in range(1, remainder + 1):
                        counts[-i] += 1
                else:
                    shot_number = 1
                    base_count = self.max_context_frames // shot_number
                    remainder = self.max_context_frames % shot_number
                    counts = [base_count] * shot_number
                    for i in range(1, remainder + 1):
                        counts[-i] += 1
            else:
                counts = [self.max_context_frames]

            condition_indices = []
            shot_flags_for_rope = []
            video_data = None
            
            if latent_gen_iter == 0:
                condition_indices += [0] * counts[0]
                shot_flags_for_rope += [0] * counts[0]
                # latent_indices = torch.where(shot_flags_gt[0]==0)
            else:
                video_data = torch.concat(output_images_list, dim=0).to(device).to(dtype)
                video_data = rearrange(video_data, 'f c h w -> f h w c')
                use_retrieval = self.semantic_retrieval_context
                if self.route_retrieval_by_prompt_similarity:
                    use_retrieval, sim = self._should_use_retrieval(
                        prev_query_embed=prev_shot_query_embed,
                        cur_query_embed=current_shot_query_embed,
                    )
                    print(
                        f"[Routing] shot={latent_gen_iter}, cosine(prev,cur)={sim:.4f}, "
                        f"use_retrieval={use_retrieval}, threshold={self.route_retrieval_cosine_threshold}"
                    )
                if use_retrieval:
                    # Retrieval query should come from current-shot prompt only
                    # (exclude global prompt to avoid retrieval bias).
                    query_text = current_shot_query
                    candidate_entries = self._build_candidate_entries(shot_flags, latent_gen_iter)
                    selected_indices, selected_shot_flags = self._kv_retrieval_topk_history(
                        history_video=video_data,
                        candidate_entries=candidate_entries,
                        query_text=query_text,
                        device=device,
                        dtype=dtype,
                    )
                    if selected_indices is not None:
                        condition_indices += selected_indices
                        shot_flags_for_rope += selected_shot_flags
                if len(condition_indices) == 0:
                    for shot_index, shot_flag in enumerate(shot_flags_unique_gt):
                        indices = torch.where(shot_flags==shot_flag)
                        if shot_flag < latent_gen_iter:
                            if self.dynamic_sample_frames:  # for dynamic sample frames
                                start_idx = min(indices[0]).item()
                                end_idx = max(indices[0]).item()
                                sampled_steps = torch.linspace(start_idx, end_idx, steps=counts[shot_index])
                                sampled_indices = sampled_steps.round().long().tolist()
                                condition_indices += sampled_indices
                                shot_flags_for_rope += [shot_index] * len(sampled_indices)
                            else:
                                condition_indices.append(min(indices[0]).item())
                                condition_indices.append(max(indices[0]).item())
                        else:
                            break

            condition_indices = torch.tensor(condition_indices, dtype=torch.int32, device=device)
            if latent_gen_iter == 0:
                # condition_frames = video_data_gt[condition_indices]  # f h w c
                condition_frames = torch.zeros([self.max_context_frames, 480, 832, 3]).to(device).to(dtype)  # Hard Code for Resolution
            else:
                condition_frames = video_data[condition_indices]  # f h w c

            if self.dynamic_sample_frames:
                assert condition_frames.shape[0] == self.max_context_frames

            # video_data = video_data_gt[latent_indices[0]]  # f h w c 

            # if self.restrict_max_length is not None:
            #     if video_data.shape[0] >= self.restrict_max_length:
            #         stride = video_data.shape[0] // self.restrict_max_length  # downsample for larger motion 
            #         video_data = video_data[ ::stride]
            #         video_data = video_data[ :self.restrict_max_length]

            # if video_data.shape[0] % 4 != 1:
            #     video_data = video_data[ :(video_data.shape[0]-1) // 4 * 4+1]

            # VAE shape: Input [b, c, f, h, w] -> [b, f, c, h, w]
            condition_frames = rearrange(condition_frames, 'f h w c -> f c 1 h w').to(device).to(dtype)  # encode each frame
            # video_data = rearrange(video_data, 'f h w c -> 1 c f h w').to(device).to(dtype)

            with torch.no_grad():
                condition_latents = self.vae.encode_to_latent(condition_frames).to(device).to(dtype)  # [f, 1, c, h, w]
                condition_latents = rearrange(condition_latents, 'f 1 c h w -> 1 f c h w')
                if latent_gen_iter == 0:
                    condition_latents = torch.zeros_like(condition_latents).to(device).to(dtype)  
                    print(f"latent_gen_iter is 0, set condition_latents as 0")
                # video_latents = self.vae.encode_to_latent(video_data)  # [1, f, c, h, w]

            if self.multi_caption:
                caption_s = []
                for i in range(latent_gen_iter+1):
                    caption = global_captions[0][0] + shots_captions[0][i][0][0]
                    caption_s.append(caption)
            else:
                caption = global_captions[0][0] + shots_captions[0][latent_gen_iter][0][0]

            # noise = torch.randn_like(video_latents)  # [1, f, c, h, w]
            noise = torch.randn([1, 21, condition_latents.shape[-3], condition_latents.shape[-2], condition_latents.shape[-1]]).to(device).to(dtype)  # [1, f, c, h, w]

            shot_flags_for_rope += [latent_gen_iter] * noise.shape[1]
            shot_flags_for_rope = torch.tensor(shot_flags_for_rope).to(torch.int32).to(device)

            batch_size, num_output_frames, num_channels, height, width = noise.shape
            
            assert num_output_frames % self.num_frame_per_block == 0
            num_blocks = num_output_frames // self.num_frame_per_block

            with torch.no_grad():
                prompts = caption_s if self.multi_caption else [caption]
                conditional_dict = self.text_encoder(
                    text_prompts=prompts,
                )

            output = torch.zeros(
                [batch_size, num_output_frames, num_channels, height, width],
                device=device,
                dtype=noise.dtype
            )

            # Step 1: Initialize KV cache to all zeros
            local_attn_cfg = getattr(self.args, "local_attn_size", -1)
            kv_policy = ""
            if local_attn_cfg != -1:
                # local attention
                kv_cache_size = local_attn_cfg * self.frame_seq_length
                kv_policy = f"int->local, size={local_attn_cfg}"
            else:
                # global attention
                kv_cache_size = num_output_frames * self.frame_seq_length
                kv_policy = "global (-1)"
            # print(f"kv_cache_size: {kv_cache_size} (policy: {kv_policy}, frame_seq_length: {self.frame_seq_length}, num_output_frames: {num_output_frames})")

            condition_frame_numbers = condition_latents.shape[1]
            kv_cache_size_context = condition_frame_numbers * self.frame_seq_length

            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
                kv_cache_size_override=kv_cache_size
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            self._initialize_context_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
                kv_cache_size_override=kv_cache_size_context
            )

            current_start_frame = 0
            self.generator.model.local_attn_size = self.local_attn_size
            print(f"[inference] local_attn_size set on model: {self.generator.model.local_attn_size}")
            self._set_all_modules_max_attention_size(self.local_attn_size)

            if torch.cuda.is_available():
                torch.cuda.synchronize(device=device)

            # Save Condition Cache
            context_timestep = torch.ones(
            [batch_size, condition_frame_numbers],
            device=noise.device,
            dtype=torch.int64) * 0

            self.generator(
                noisy_image_or_video=condition_latents,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=None,  # When KV Cache is None, Context KV Cache and CrossAtten Cache is not None
                crossattn_cache=self.crossattn_cache,
                ###
                kv_cache_context=self.kv_cache1_context,
                shot_flags_for_rope = shot_flags_for_rope[:condition_latents.shape[1]] if self.change_rope else None,
                ###
                current_start=0,
            )

            # Step 2: Temporal denoising loop
            all_num_frames = [self.num_frame_per_block] * num_blocks
            for current_num_frames in all_num_frames:

                noisy_input = noise[
                    :, current_start_frame:current_start_frame + current_num_frames]

                # Step 2.1: Spatial denoising loop
                for index, current_timestep in enumerate(self.denoising_step_list):
                    print(f"timestep is {current_timestep}")
                    # set current timestep
                    timestep = torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64) * current_timestep

                    if index < len(self.denoising_step_list) - 1:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            ###
                            kv_cache_context=self.kv_cache1_context,
                            shot_flags_for_rope = shot_flags_for_rope[condition_latents.shape[1]+current_start_frame: condition_latents.shape[1]+current_start_frame+noisy_input.shape[1]]  if self.change_rope else None,
                            use_wo_rope_cache=use_wo_rope_cache,
                            ###
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                    else:
                        # for getting real output
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            ###
                            kv_cache_context=self.kv_cache1_context,
                            shot_flags_for_rope = shot_flags_for_rope[condition_latents.shape[1]+current_start_frame: condition_latents.shape[1]+current_start_frame+noisy_input.shape[1]] if self.change_rope else None,
                            use_wo_rope_cache=use_wo_rope_cache,
                            ###
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length
                        )
                # Step 2.2: record the model's output
                output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred.to(output.device)
                # Step 2.3: rerun with timestep zero to update KV cache using clean context
                context_timestep = torch.ones_like(timestep) * self.args.context_noise
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    ###
                    kv_cache_context=self.kv_cache1_context,
                    shot_flags_for_rope = shot_flags_for_rope[condition_latents.shape[1]+current_start_frame: condition_latents.shape[1]+current_start_frame+noisy_input.shape[1]] if self.change_rope else None,
                    use_wo_rope_cache=use_wo_rope_cache,
                    ###
                )

                # Step 3.4: update the start and end frame indices
                current_start_frame += current_num_frames

            if torch.cuda.is_available():
                torch.cuda.synchronize(device=device)

            # Step 3: Decode the output
            video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
            output_images_list.append(video[0])
            shot_flags_output += [latent_gen_iter] * video.shape[1]
            prev_shot_query_embed = current_shot_query_embed.detach()
            print(f"shot {latent_gen_iter} end")

        video = torch.concat(output_images_list, dim=0)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        
        return video[None]

    def _initialize_kv_cache(self, batch_size, dtype, device, kv_cache_size_override: int | None = None):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        # Determine cache size
        if kv_cache_size_override is not None:
            kv_cache_size = kv_cache_size_override
        else:
            if self.local_attn_size != -1:
                # Local attention: cache only needs to store the window
                kv_cache_size = self.local_attn_size * self.frame_seq_length
            else:
                # Global attention: default cache for 21 frames (backward compatibility)
                kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                # "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                # "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "k": torch.zeros([batch_size, kv_cache_size, self.num_heads, self.d], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, self.num_heads, self.d], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_context_kv_cache(self, batch_size, dtype, device, kv_cache_size_override: int | None = None):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1_context = []
        # Determine cache size
        kv_cache_size = kv_cache_size_override

        for _ in range(self.num_transformer_blocks):
            kv_cache1_context.append({
                # "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                # "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "k": torch.zeros([batch_size, kv_cache_size, self.num_heads, self.d], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, self.num_heads, self.d], dtype=dtype, device=device),
                "is_init": False,
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1_context = kv_cache1_context  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                # "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                # "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "k": torch.zeros([batch_size, 512, self.num_heads, self.d], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, self.num_heads, self.d], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int):
        """
        Set max_attention_size on all submodules that define it.
        If local_attn_size_value == -1, use the model's global default (32760 for Wan, 28160 for 5B).
        Otherwise, set to local_attn_size_value * frame_seq_length.
        """
        if local_attn_size_value == -1:
            target_size = 32760
            policy = "global"
        else:
            target_size = int(local_attn_size_value) * self.frame_seq_length
            policy = "local"

        updated_modules = []
        # Update root model if applicable
        if hasattr(self.generator.model, "max_attention_size"):
            try:
                prev = getattr(self.generator.model, "max_attention_size")
            except Exception:
                prev = None
            setattr(self.generator.model, "max_attention_size", target_size)
            updated_modules.append("<root_model>")

        # Update all child modules
        for name, module in self.generator.model.named_modules():
            if hasattr(module, "max_attention_size"):
                try:
                    prev = getattr(module, "max_attention_size")
                except Exception:
                    prev = None
                try:
                    setattr(module, "max_attention_size", target_size)
                    updated_modules.append(name if name else module.__class__.__name__)
                except Exception:
                    pass


    def _configure_lora_for_model(self, transformer, model_name):
        """Configure LoRA for a WanDiffusionWrapper model"""
        # Find all Linear modules in WanAttentionBlock modules
        target_linear_modules = set()
        
        # Define the specific modules we want to apply LoRA to
        if model_name == 'generator':
            adapter_target_modules = ['CausalWanAttentionBlock']
        elif model_name == 'fake_score':
            adapter_target_modules = ['WanAttentionBlock']
        else:
            raise ValueError(f"Invalid model name: {model_name}")
        
        for name, module in transformer.named_modules():
            if module.__class__.__name__ in adapter_target_modules:
                for full_submodule_name, submodule in module.named_modules(prefix=name):
                    if isinstance(submodule, torch.nn.Linear):
                        target_linear_modules.add(full_submodule_name)
        
        target_linear_modules = list(target_linear_modules)
        
        # if self.is_main_process:
        print(f"LoRA target modules for {model_name}: {len(target_linear_modules)} Linear layers")
        if getattr(self.lora_config, 'verbose', False):
            for module_name in sorted(target_linear_modules):
                print(f"  - {module_name}")
        
        # Create LoRA config
        adapter_type = self.lora_config.get('type', 'lora')
        if adapter_type == 'lora':
            peft_config = peft.LoraConfig(
                r=self.lora_config.get('rank', 16),
                lora_alpha=self.lora_config.get('alpha', None) or self.lora_config.get('rank', 16),
                lora_dropout=self.lora_config.get('dropout', 0.0),
                target_modules=target_linear_modules,
                # task_type="FEATURE_EXTRACTION"        # Remove this; not needed for diffusion models
            )
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')
        
        # Apply LoRA to the transformer
        lora_model = peft.get_peft_model(transformer, peft_config)

        # if self.is_main_process:
        print('peft_config', peft_config)
        lora_model.print_trainable_parameters()

        return lora_model
