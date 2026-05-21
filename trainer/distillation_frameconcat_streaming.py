# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import gc
import logging
import random

from utils.dataset import MultiShots_FrameConcat_Dataset, cycle
from utils.distributed import fsdp_wrap, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from model import DMDFrameConcat
from model.streaming_training import StreamingTrainingModel
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
import os
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    StateDictType, FullStateDictConfig, FullOptimStateDictConfig
)

import peft
from peft import get_peft_model_state_dict

from utils.memory import log_gpu_memory

from utils.debug_option import LOG_GPU_MEMORY
import time
from einops import rearrange

class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        # MODIFIED: Changed disable_wandb to disable_tensorboard
        self.disable_tensorboard = config.disable_tensorboard 

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)
        print(f"set the seed as {config.seed} + global_rank")

        self.use_one_logger = getattr(config, "use_one_logger", False)
        self.dmd_debug = getattr(config, "dmd_debug", False)
        self.max_context_frames = getattr(config, "max_context_frames", 10)  # for dynamic sample frames
        self.dynamic_sample_frames = getattr(config, "dynamic_sample_frames", False)
        self.change_rope = getattr(config, "change_rope", False)
        self.train_lora = getattr(config, "train_lora", False)
        self.train_lora_generator = getattr(config, "train_lora_generator", False)
        self.multi_caption = getattr(config, "multi_caption", False)  # for multi caption
        self.semantic_retrieval_context = getattr(config, "semantic_retrieval_context", False)
        self.semantic_memory_frames_per_shot = getattr(config, "semantic_memory_frames_per_shot", 1)
        self.context_rope_max_id = getattr(config, "context_rope_max_id", 2)

        self.train_lora_fake = getattr(config, "train_lora_fake", False)
        print(f"self.train_lora_fake is {self.train_lora_fake}")

        self.real_fake_use_gt_context = getattr(config, "real_fake_use_gt_context", False)

        self.output_path = config.logdir
        
        if self.is_main_process:
            import datetime
            current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self.output_path = os.path.join(self.output_path, f"{config.log_name}-{current_time}")
            os.makedirs(self.output_path, exist_ok=True)

        # MODIFIED: Replaced wandb setup with TensorBoard SummaryWriter
        self.writer = None
        if self.is_main_process and not self.disable_tensorboard:
            print(f"Initializing TensorBoard SummaryWriter in {self.output_path}")
            self.writer = SummaryWriter(log_dir=self.output_path)

        # Step 2: Initialize the model
        log_gpu_memory("Before initialization Model", device=self.device, rank=dist.get_rank())
        self.model = DMDFrameConcat(config, device=self.device)
        self.model.requires_grad_(False)
        log_gpu_memory("After initialization Model", device=self.device, rank=dist.get_rank())

        # Lora setting
        lora_path = getattr(self.config, "resume_ckpt_lora", None)
        real_score_offload_params = getattr(self.config, "real_score_offload_params", False)
        has_lora_file = lora_path and os.path.exists(lora_path)
        self.lora_config = None
        if hasattr(config, 'adapter') and config.adapter is not None:
            self.lora_config = config.adapter

        if self.is_main_process:
            print(f"Begin Load Ckpt!")

        # Load Generator
        # log_gpu_memory("Before Load Generator", device=self.device, rank=dist.get_rank())
        if self.config.resume_ckpt is not None:
            print(f"[Generator] Rank 0 loading base ckpt: {self.config.resume_ckpt}")
            ckpt = torch.load(self.config.resume_ckpt, map_location='cpu', mmap=True, weights_only=False)['generator']
            self.model.generator.load_state_dict(ckpt)
            print(f"[Generator Done] Rank 0 loading base ckpt: {self.config.resume_ckpt}")
            del ckpt
            gc.collect()
        # log_gpu_memory("After Load Generator", device=self.device, rank=dist.get_rank())
        
        # log_gpu_memory("Before Set Lora Generator", device=self.device, rank=dist.get_rank())
        if self.train_lora or self.train_lora_generator:
            self.model.generator.model = self._configure_lora_for_model(self.model.generator.model, "generator")
            print(f"[Generator Done] set lora")
        # log_gpu_memory("After Set Lora Generator", device=self.device, rank=dist.get_rank())
        
        if (self.train_lora or self.train_lora_generator) and has_lora_file:
            if self.is_main_process:
                print(f"[Generator ] Rank 0 loading LoRA weights...")
                lora_checkpoint = torch.load(lora_path, map_location="cpu", mmap=True)
                if "generator_lora" in lora_checkpoint:
                    peft.set_peft_model_state_dict(self.model.generator.model, lora_checkpoint["generator_lora"])
                del lora_checkpoint; gc.collect()
            print(f"[Generator Done] Rank 0 loading LoRA weights...")
        
        # log_gpu_memory("Before FSDP Generator", device=self.device, rank=dist.get_rank())
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            sync_module_states=True,  # 广播 Rank 0 的所有权重
            min_num_params=int(1e7),
        )
        if dist.is_initialized(): dist.barrier()
        print(f"[Generator Done] syn to all ranks done")
        # log_gpu_memory("After FSDP Generator", device=self.device, rank=dist.get_rank())

        # Load Real Score Model
        # log_gpu_memory("Before Load Real Ckpt", device=self.device, rank=dist.get_rank())
        if self.config.resume_ckpt_real is not None:
            if self.is_main_process:
                print(f"[Real Score] Rank 0 loading base ckpt: {self.config.resume_ckpt_real}")
                ckpt = torch.load(self.config.resume_ckpt_real, map_location='cpu', mmap=True, weights_only=False)['generator']
                self.model.real_score.load_state_dict(ckpt)
                del ckpt; gc.collect()
                print(f"[Real Score Done] Rank 0 loading base ckpt: {self.config.resume_ckpt_real}")
        # log_gpu_memory("After Load Real Ckpt", device=self.device, rank=dist.get_rank())
        
        # log_gpu_memory("Before FSDP Real", device=self.device, rank=dist.get_rank())
        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            sync_module_states=True,
            cpu_offload=real_score_offload_params,
            min_num_params=int(1e7),
        )
        # log_gpu_memory("After FSDP Real", device=self.device, rank=dist.get_rank())
        if dist.is_initialized(): dist.barrier()
        print(f"[Real Score Done] syn to all ranks done")

        # Fake Score Model
        if self.config.resume_ckpt_fake is not None:
            if self.is_main_process:
                print(f"[Fake Score] Rank 0 loading base ckpt: {self.config.resume_ckpt_fake}")
                ckpt = torch.load(self.config.resume_ckpt_fake, map_location='cpu', mmap=True, weights_only=False)['generator']
                self.model.fake_score.load_state_dict(ckpt)
                del ckpt; gc.collect()
                print(f"[Fake Score Done] Rank 0 loading base ckpt: {self.config.resume_ckpt_fake}")
        
        if (self.train_lora and getattr(config.adapter, 'apply_to_critic', True)) or self.train_lora_fake:
            self.model.fake_score.model = self._configure_lora_for_model(self.model.fake_score.model, "fake_score")
            print(f"[Fake Score Done] set lora")

        if ((self.train_lora and getattr(config.adapter, 'apply_to_critic', True)) or self.train_lora_fake) and has_lora_file:
            if self.is_main_process:
                print(f"[Fake Score] Rank 0 loading LoRA weights...")
                lora_checkpoint = torch.load(lora_path, map_location="cpu", mmap=True)
                if "critic_lora" in lora_checkpoint:
                    peft.set_peft_model_state_dict(self.model.fake_score.model, lora_checkpoint["critic_lora"])
                del lora_checkpoint; gc.collect()
            print(f"[Fake Score Done] Rank 0 loading LoRA weights...")

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            sync_module_states=True,
            min_num_params=int(1e7),
        )
        if dist.is_initialized(): dist.barrier()
        print(f"[Fake Score Done] syn to all ranks done")

        if self.is_main_process:
            print(f"Done Load Ckpt!")

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False),
            min_num_params=int(1e7),
        )
        self.model.vae = self.model.vae.to(
            device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        # Step 3: Set Trainable Parameters
        if (not self.train_lora) and ((not self.train_lora_generator) or (not self.train_lora_fake)):
            print("Freezing all model parameters...")
            print("Unfreezing generator parameters...")
            # self.model.generator.requires_grad_(True)
            if config.only_train3d is True:
                if not self.train_lora_generator:
                    for name, module in self.model.generator.named_modules():
                        if any(keyword in name for keyword in ["self_attn"]):
                            print(f"Trainable: {name}")
                            for param in module.parameters():
                                param.requires_grad = True
                if not self.train_lora_fake:
                    for name, module in self.model.fake_score.named_modules():
                        if any(keyword in name for keyword in ["self_attn"]):
                            print(f"Trainable: {name}")
                            for param in module.parameters():
                                param.requires_grad = True
            else:
                if not self.train_lora_generator:
                    self.model.generator.requires_grad_(True)
                if not self.train_lora_fake:
                    self.model.fake_score.requires_grad_(True)

        print("\nVerifying parameter grad status:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"  - [TRAINABLE] {name}")
        
        # Step 4: Set Optimizer
        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 5: Set Dataset
        dataset = MultiShots_FrameConcat_Dataset(config.data_path)
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=False)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            collate_fn=dataset.custom_collate_fn,
            num_workers=0)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        # Step 6: Set default param for gradient clip
        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        self.previous_time = None
        
        # streaming training configuration
        self.streaming_training = getattr(config, "streaming_training", False)
        self.streaming_chunk_size = getattr(config, "streaming_chunk_size", 21)
        self.streaming_max_length = getattr(config, "streaming_max_length", 63)
        
        # Create streaming training model if enabled
        if self.streaming_training:
            self.streaming_model = StreamingTrainingModel(self.model, config)
            if self.is_main_process:
                print(f"streaming training enabled: chunk_size={self.streaming_chunk_size}, max_length={self.streaming_max_length}")
        else:
            self.streaming_model = None
        
        self.streaming_active = False  # Whether we're currently in a sequence
        
        if self.is_main_process:
            print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
            if self.gradient_accumulation_steps > 1:
                print(f"Effective batch size: {config.batch_size * self.gradient_accumulation_steps * self.world_size}")
            if self.streaming_training:
                print(f"streaming training enabled: chunk_size={self.streaming_chunk_size}, max_length={self.streaming_max_length}")
            # if LOG_GPU_MEMORY:
            #     log_gpu_memory("After initialization", device=self.device, rank=dist.get_rank())

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
        
        if self.is_main_process:
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

        if self.is_main_process:
            print('peft_config', peft_config)
            lora_model.print_trainable_parameters()

        return lora_model

    def _gather_lora_state_dict(self, lora_model):
        "On rank-0, gather FULL_STATE_DICT, then filter only LoRA weights"
        with FSDP.state_dict_type(
            lora_model,                       # lora_model contains nested FSDP submodules
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
        ):
            full = lora_model.state_dict()
        return get_peft_model_state_dict(lora_model, state_dict=full)

    def _broadcast_model_weights(self, model):
        """将 Rank 0 的模型权重广播到所有其他进程"""
        if not dist.is_initialized():
            return
        
        # 同步所有参数 (Parameters)
        for param in model.parameters():
            dist.broadcast(param.data, src=0)
        
        # 同步所有缓冲层 (Buffers, 如 BatchNorm 的 running_mean 等)
        for buffer in model.buffers():
            dist.broadcast(buffer.data, src=0)


    def _get_switch_frame_index(self, max_length=None):
        if getattr(self.config, "switch_mode", "fixed") == "random":
            block = self.config.num_frame_per_block
            min_idx = self.config.min_switch_frame_index
            max_idx = self.config.max_switch_frame_index
            if min_idx == max_idx:
                switch_idx = min_idx
            else:
                choices = list(range(min_idx, max_idx, block))
                if max_length is not None:
                    choices = [choice for choice in choices if choice < max_length]
                
                if len(choices) == 0:
                    if max_length is not None:
                        raise ValueError(f"No valid switch choices available (all choices >= max_length {max_length})")
                    else:
                        switch_idx = block
                else:
                    if dist.get_rank() == 0:
                        switch_idx = random.choice(choices)
                    else:
                        switch_idx = 0  # placeholder; will be overwritten by broadcast
                switch_idx_tensor = torch.tensor(switch_idx, device=self.device)
                dist.broadcast(switch_idx_tensor, src=0)
                switch_idx = switch_idx_tensor.item()
        elif getattr(self.config, "switch_mode", "fixed") == "fixed":
            switch_idx = getattr(self.config, "fixed_switch_index", 21)
            if max_length is not None:
                assert max_length > switch_idx, f"max_length {max_length} is not greater than switch_idx {switch_idx}"
        elif getattr(self.config, "switch_mode", "fixed") == "random_choice":
            switch_choices = getattr(self.config, "switch_choices", [])
            if len(switch_choices) == 0:
                raise ValueError("switch_choices is empty")
            else:
                if max_length is not None:
                    switch_choices = [choice for choice in switch_choices if choice < max_length]
                    if len(switch_choices) == 0:
                        raise ValueError(f"No valid switch choices available (all choices >= max_length {max_length})")
                
                if dist.get_rank() == 0:
                    switch_idx = random.choice(switch_choices)
                else:
                    switch_idx = 0
            switch_idx_tensor = torch.tensor(switch_idx, device=self.device)
            dist.broadcast(switch_idx_tensor, src=0)
            switch_idx = switch_idx_tensor.item()
        else:
            raise ValueError(f"Invalid switch_mode: {getattr(self.config, 'switch_mode', 'fixed')}")
        return switch_idx


    def save(self):
        print("Start gathering distributed model states...")

        if self.train_lora:
            gen_lora_sd = self._gather_lora_state_dict(
                self.model.generator.model)
            crit_lora_sd = self._gather_lora_state_dict(
                self.model.fake_score.model)

            state_dict = {
                "generator_lora": gen_lora_sd,
                "critic_lora": crit_lora_sd,
                "step": self.step,
            }
        elif (not self.train_lora_generator) and (not self.train_lora_fake) and (not self.train_lora):
            with FSDP.state_dict_type(
                self.model.generator,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                FullOptimStateDictConfig(rank0_only=True),          # newly added
            ):
                generator_state_dict  = self.model.generator.state_dict()
                generator_opim_state_dict = FSDP.optim_state_dict(self.model.generator,
                                                self.generator_optimizer)

            with FSDP.state_dict_type(
                self.model.fake_score,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                FullOptimStateDictConfig(rank0_only=True),          # newly added
            ):
                fake_score_state_dict  = self.model.fake_score.state_dict()
                fake_score_opim_state_dict = FSDP.optim_state_dict(self.model.fake_score,
                                                self.critic_optimizer)

            state_dict = {
                "generator": generator_state_dict,
                "generator_optimizer": generator_opim_state_dict,
                "step": self.step,
                "fake_score": fake_score_state_dict,
                "critic_optimizer": fake_score_opim_state_dict,
            }

        else:
            state_dict = {"step": self.step,}
            if self.train_lora_generator:
                gen_lora_sd = self._gather_lora_state_dict(
                self.model.generator.model)
                state_dict.update({"generator_lora": gen_lora_sd})

            else:
                with FSDP.state_dict_type(
                    self.model.generator,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                    FullOptimStateDictConfig(rank0_only=True),          # newly added
                ):
                    generator_state_dict  = self.model.generator.state_dict()
                    generator_opim_state_dict = FSDP.optim_state_dict(self.model.generator,
                                                    self.generator_optimizer)
                state_dict.update({"generator": generator_state_dict,
                "generator_optimizer": generator_opim_state_dict,})
                
            if self.train_lora_fake:
                crit_lora_sd = self._gather_lora_state_dict(
                self.model.fake_score.model)
                state_dict.update({"critic_lora": crit_lora_sd,})
            else:
                with FSDP.state_dict_type(
                    self.model.fake_score,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
                    FullOptimStateDictConfig(rank0_only=True),          # newly added
                ):
                    fake_score_state_dict  = self.model.fake_score.state_dict()
                    fake_score_opim_state_dict = FSDP.optim_state_dict(self.model.fake_score,
                                                    self.critic_optimizer)
                state_dict.update({"fake_score": fake_score_state_dict,
                "critic_optimizer": fake_score_opim_state_dict,})

        if self.is_main_process:  # Only save in main process
            checkpoint_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_file = os.path.join(checkpoint_dir, "model.pt")
            torch.save(state_dict, checkpoint_file)
            print("Model saved to", checkpoint_file)
            
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    def _shot_prompt(self, global_captions, shots_captions, shot_index: int, include_global: bool = False):
        shot_prompt = shots_captions[0][0][shot_index][0]
        if include_global:
            return global_captions[0][0] + shot_prompt
        return shot_prompt

    def _to_local_context_flags(self, shot_flags_list):
        unique_sorted = sorted(set(int(x) for x in shot_flags_list))
        id_map = {sid: i for i, sid in enumerate(unique_sorted)}
        return [id_map[int(x)] for x in shot_flags_list]

    def _to_small_context_flags(self, shot_flags_list):
        local_flags = self._to_local_context_flags(shot_flags_list)
        max_id = max(0, int(self.context_rope_max_id))
        return [min(int(v), max_id) for v in local_flags]

    @torch.no_grad()
    def _select_semantic_context_indices_streaming(self, shot_flags, latent_gen_iter, global_captions, shots_captions):
        if len(shot_flags) == 0 or latent_gen_iter <= 0:
            return None, None

        candidate_entries = []
        for shot_id in range(latent_gen_iter):
            indices = torch.where(shot_flags == shot_id)[0].tolist()
            for idx in indices[:self.semantic_memory_frames_per_shot]:
                candidate_entries.append((idx, shot_id))

        if len(candidate_entries) == 0:
            return None, None

        query_prompt = self._shot_prompt(global_captions, shots_captions, latent_gen_iter, include_global=False)
        memory_prompts = [
            self._shot_prompt(global_captions, shots_captions, shot_id, include_global=False)
            for _, shot_id in candidate_entries
        ]
        prompt_embeds = self.model.text_encoder(text_prompts=[query_prompt] + memory_prompts)["prompt_embeds"].float()
        prompt_embeds = F.normalize(prompt_embeds.mean(dim=1), dim=-1)
        scores = prompt_embeds[1:] @ prompt_embeds[0]
        top_k = min(self.max_context_frames, scores.shape[0])
        top_ids = torch.topk(scores, k=top_k, dim=0).indices.tolist()

        selected_indices = [candidate_entries[i][0] for i in top_ids]
        selected_flags = [candidate_entries[i][1] for i in top_ids]
        selected_flags = self._to_small_context_flags(selected_flags)
        if len(selected_indices) < self.max_context_frames and len(selected_indices) > 0:
            pad_num = self.max_context_frames - len(selected_indices)
            selected_indices += [selected_indices[-1]] * pad_num
            selected_flags += [selected_flags[-1]] * pad_num
        return selected_indices, selected_flags

    def fwdbwd_one_step(self, batch, train_generator, latent_gen_iter=0, **kwargs):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        device, dtype = self.model.vae.model.encoder.conv1.weight.device, self.model.vae.model.encoder.conv1.weight.dtype 

        if self.step % 50 == 0:
            torch.cuda.empty_cache()

        # Step 1: Prepare data
        data_path = batch['data_path']
        video_data_gt = torch.tensor(batch['data'][0])  # [f h w c]
        global_captions = batch['global_captions']
        shots_captions = batch['shots_captions']
        shot_flags_gt = torch.tensor([batch['shot_flag']]).to(torch.int32) 
        shot_flags_unique_gt = torch.unique(shot_flags_gt)

        shot_flags_output = kwargs['shot_flags_output']
        output_images_list = kwargs['output_images_list']

        shot_flags = torch.tensor(shot_flags_output).to(torch.int32) 
        shot_flags_unique = torch.unique(shot_flags)

        if self.dynamic_sample_frames:
            if latent_gen_iter > 0:
                shot_number = shot_flags_unique.shape[0]  # except the noise latent 
                base_count = self.max_context_frames // shot_number
                remainder = self.max_context_frames % shot_number
                counts = [base_count] * shot_number
                for i in range(1, remainder + 1):
                    counts[-i] += 1
            elif latent_gen_iter == 0:
                shot_number = 1
                base_count = self.max_context_frames // shot_number
                remainder = self.max_context_frames % shot_number
                counts = [base_count] * shot_number
                for i in range(1, remainder + 1):
                    counts[-i] += 1

        # print(f"[DEBUG] Gen iter {latent_gen_iter}, counts is {counts}")

        condition_indices = []
        shot_flags_for_rope = []
        condition_indices_gt = []  # for gt context

        if latent_gen_iter == 0:
            condition_indices += [0] * counts[0]
            shot_flags_for_rope += [0] * counts[0]
            latent_indices = torch.where(shot_flags_gt[0]==0)
            if self.real_fake_use_gt_context:
                condition_indices_gt += [0] * counts[0]
        else:
            if self.semantic_retrieval_context:
                selected_indices, selected_flags = self._select_semantic_context_indices_streaming(
                    shot_flags=shot_flags,
                    latent_gen_iter=latent_gen_iter,
                    global_captions=global_captions,
                    shots_captions=shots_captions,
                )
                if selected_indices is not None:
                    condition_indices += selected_indices
                    shot_flags_for_rope += selected_flags

            if len(condition_indices) == 0:
                for shot_index, shot_flag in enumerate(shot_flags_unique_gt):
                    indices = torch.where(shot_flags==shot_flag)
                    # if shot_flag != max(shot_flags_unique):
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
                    elif shot_flag == latent_gen_iter:
                        indices = torch.where(shot_flags_gt[0] == shot_flag)
                        latent_indices = indices
                    else:
                        break
                    
            if self.real_fake_use_gt_context:
                for shot_index, shot_flag in enumerate(shot_flags_unique_gt):
                    indices = torch.where(shot_flags_gt[0][0]==shot_flag)
                    # if shot_flag != max(shot_flags_unique):
                    if shot_flag < latent_gen_iter:
                        if self.dynamic_sample_frames:  # for dynamic sample frames
                            start_idx = min(indices[0]).item()
                            end_idx = max(indices[0]).item()
                            sampled_steps = torch.linspace(start_idx, end_idx, steps=counts[shot_index])
                            sampled_indices = sampled_steps.round().long().tolist()
                            condition_indices_gt += sampled_indices
                        else:
                            condition_indices_gt.append(min(indices[0]).item())
                            condition_indices_gt.append(max(indices[0]).item())
                    elif shot_flag == latent_gen_iter:
                        indices = torch.where(shot_flags_gt[0] == shot_flag)
                        latent_indices = indices
                    else:
                        break

        condition_indices = torch.tensor(condition_indices, dtype=torch.int32, device=video_data_gt.device)

        if self.real_fake_use_gt_context:
            condition_indices_gt = torch.tensor(condition_indices_gt, dtype=torch.int32, device=video_data_gt.device)
        
        if latent_gen_iter == 0:
            condition_frames = video_data_gt[condition_indices]  # f h w c
            if self.real_fake_use_gt_context:
                condition_frames_gt = video_data_gt[condition_indices_gt]  # f h w c
        else:
            video_data = torch.concat(output_images_list, dim=0).to(device).to(dtype)
            video_data = rearrange(video_data, 'f c h w -> f h w c')
            condition_frames = video_data[condition_indices]  # f h w c
            if self.real_fake_use_gt_context:
                condition_frames_gt = video_data_gt[condition_indices_gt]  # f h w c
            
        if self.dynamic_sample_frames:
            assert condition_frames.shape[0] == self.max_context_frames

        video_data = video_data_gt[latent_indices[0]]  # f h w c 
        if self.config.restrict_max_length is not None:
            if video_data.shape[0] >= self.config.restrict_max_length:
                stride = video_data.shape[0] // self.config.restrict_max_length  # downsample for larger motion 
                video_data = video_data[ ::stride]
                video_data = video_data[ :self.config.restrict_max_length]

        if video_data.shape[0] % 4 != 1:
            video_data = video_data[ :(video_data.shape[0]-1) // 4 * 4+1]

        # VAE shape: Input [b, c, f, h, w] -> [b, f, c, h, w]
        condition_frames = rearrange(condition_frames, 'f h w c -> f c 1 h w').to(device).to(dtype)  # encode each frame
        if self.real_fake_use_gt_context:
            condition_frames_gt = rearrange(condition_frames_gt, 'f h w c -> f c 1 h w').to(device).to(dtype)
        video_data = rearrange(video_data, 'f h w c -> 1 c f h w').to(device).to(dtype)

        with torch.no_grad():
            condition_latents = self.model.vae.encode_to_latent(condition_frames)  # [f, 1, c, h, w]
            if self.real_fake_use_gt_context:
                condition_latents_gt = self.model.vae.encode_to_latent(condition_frames_gt)  # [f, 1, c, h, w]
            if latent_gen_iter == 0:
                condition_latents = torch.zeros_like(condition_latents).to(device).to(dtype)  
                if self.real_fake_use_gt_context:
                    condition_latents_gt = torch.zeros_like(condition_latents_gt).to(device).to(dtype)

            import random
            if random.random() < 0.1:  # [TODO]
                condition_latents = torch.zeros_like(condition_latents)
                if self.real_fake_use_gt_context:
                    condition_latents_gt = torch.zeros_like(condition_latents_gt)
                shot_flags_for_rope = [0] * len(shot_flags_for_rope)  # [Note] Also set flag as zero
            condition_latents = rearrange(condition_latents, 'f 1 c h w -> 1 f c h w')
            if self.real_fake_use_gt_context:
                condition_latents_gt = rearrange(condition_latents_gt, 'f 1 c h w -> 1 f c h w')
            video_latents = torch.randn([condition_latents.shape[0], 21, condition_latents.shape[2], condition_latents.shape[-2], condition_latents.shape[-1]])  # [1, f, c, h, w]
            
            # #### [NOTE] Hard code: Cut latents for training
            # if video_latents.shape[1]>18:
            #     video_latents = video_latents[::, :18]

        if self.dmd_debug:
            condition_latents = torch.zeros([condition_latents.shape[0], 0, condition_latents.shape[-3], condition_latents.shape[-2], condition_latents.shape[-1]]).to(video_latents.dtype).to(video_latents.device)

        if self.multi_caption:
            caption_s = []
            for i in range(len(shots_captions[0][0])):
                caption = global_captions[0][0] + shots_captions[0][0][i][0]
                caption_s.append(caption)
        else:
            caption = global_captions[0][0] + shots_captions[0][0][-1][0]

        num_rope_latent_frames = max(
            video_latents.shape[1],
            int(getattr(self.model, "num_training_frames", getattr(self.config, "num_training_frames", video_latents.shape[1])))
        )
        current_rope_flag = (max(shot_flags_for_rope) + 1) if len(shot_flags_for_rope) > 0 else 0
        shot_flags_for_rope += [current_rope_flag] * num_rope_latent_frames

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            prompts = caption_s if self.multi_caption else [caption]
            conditional_dict = self.model.text_encoder(
                text_prompts=prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * 1)  # [NOTE] Hard Code, only can deal with batchsize=1
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        shot_flags_for_rope = torch.tensor(shot_flags_for_rope).to(torch.int32).to(device)
        
        image_or_video_shape = list(video_latents.shape)  # [b, f, c, h, w]

        # print(f"[DEBUG] shot_flags for rope is {shot_flags_for_rope}")
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=None,
                initial_latent=None,
                context_latent=condition_latents,  # condition latents, which contains previous shots' frames
                shot_flags_for_rope = shot_flags_for_rope if self.change_rope else None,
                return_generator_pred_x0=True,  # add for return pred x0
                real_fake_use_gt_context = self.real_fake_use_gt_context,
                context_latent_gt=condition_latents_gt if self.real_fake_use_gt_context else None,
            )

            # Scale loss for gradient accumulation and backward
            scaled_generator_loss = generator_loss / self.gradient_accumulation_steps
            scaled_generator_loss.backward()
            if LOG_GPU_MEMORY:
                log_gpu_memory("After train_generator backward pass", device=self.device, rank=dist.get_rank())
            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": torch.tensor(0.0, device=self.device)})  # Will be computed after accumulation

            return generator_log_dict
        else:
            generator_log_dict = {}

            critic_loss, critic_log_dict = self.model.critic_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=None,
                initial_latent=None,
                context_latent=condition_latents,  # condition latents, which contains previous shots' frames
                shot_flags_for_rope = shot_flags_for_rope if self.change_rope else None,
                return_generator_pred_x0=True,  # add for return pred x0
                real_fake_use_gt_context = self.real_fake_use_gt_context,
                context_latent_gt=condition_latents_gt if self.real_fake_use_gt_context else None,
            )

            # Scale loss for gradient accumulation and backward
            scaled_critic_loss = critic_loss / self.gradient_accumulation_steps
            scaled_critic_loss.backward()
            if LOG_GPU_MEMORY:
                log_gpu_memory("After train_critic backward pass", device=self.device, rank=dist.get_rank())
            # Return original loss for logging
            critic_log_dict.update({"critic_loss": critic_loss,
                                    "critic_grad_norm": torch.tensor(0.0, device=self.device)})  # Will be computed after accumulation

            return critic_log_dict
    
    def train(self):
        start_step = self.step
        try:
            while True:
                
                # Whole-cycle gradient accumulation loop
                accumulated_generator_logs = []
                accumulated_critic_logs = []
                
                # for accumulation_step in range(self.gradient_accumulation_steps):  # [NOTE] No gradient accumulation
                batch = next(self.dataloader)

                output_images_list = []
                shot_flags_output = []
                
                for i in range(len(batch['shots_captions'][0][0])):
                    
                    # Check if we should train generator on this optimization step
                    TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
                    # print(f"[DEBUG] self.step is {self.step}, TRAIN_GENERATOR is {TRAIN_GENERATOR}")

                    if TRAIN_GENERATOR:
                        self.generator_optimizer.zero_grad(set_to_none=True)
                    else:
                        self.critic_optimizer.zero_grad(set_to_none=True)

                    kwargs={
                    "output_images_list": output_images_list,
                    "shot_flags_output": shot_flags_output,
                    }

                    # Train generator (if needed)
                    if TRAIN_GENERATOR:
                        extra_gen = self.fwdbwd_one_step(batch, True, latent_gen_iter=i, **kwargs)
                        accumulated_generator_logs.append(extra_gen)
                    else:
                        # Train critic
                        extra_crit = self.fwdbwd_one_step(batch, False, latent_gen_iter=i, **kwargs)
                        accumulated_critic_logs.append(extra_crit)
                        
                    # Compute grad norm and update parameters
                    if TRAIN_GENERATOR:
                        generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)
                        generator_log_dict = merge_dict_list(accumulated_generator_logs)
                        generator_log_dict["generator_grad_norm"] = generator_grad_norm
                        
                        self.generator_optimizer.step()
                    else:
                        generator_log_dict = {}
                    
                        critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)
                        critic_log_dict = merge_dict_list(accumulated_critic_logs)
                        critic_log_dict["critic_grad_norm"] = critic_grad_norm
                        
                        self.critic_optimizer.step()

                    # add condition list
                    # Vis for debug
                    if TRAIN_GENERATOR:
                        generator_pred_x0 = extra_gen['generator_pred_x0']
                    else:
                        generator_pred_x0 = extra_crit['generator_pred_x0']
                    
                    with torch.no_grad():
                        generator_pred_x0_videos = self.model.vae.decode_to_pixel(generator_pred_x0)
                    
                    output_images_list.append(generator_pred_x0_videos[0])
                    shot_flags_output += [i] * generator_pred_x0_videos.shape[1]
                    
                    # Increment the step since we finished gradient update
                    self.step += 1
                    current_time = time.time()
                    torch.cuda.empty_cache()

                    # Save the model
                    if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                        torch.cuda.empty_cache()
                        self.save()
                        torch.cuda.empty_cache()

                    # Logging
                    if self.is_main_process and self.writer:
                        if TRAIN_GENERATOR and generator_log_dict:
                            self.writer.add_scalar("Loss/generator", generator_log_dict["generator_loss"].mean().item(), self.step)
                            self.writer.add_scalar("GradNorm/generator", generator_log_dict["generator_grad_norm"].mean().item(), self.step)
                            self.writer.add_scalar("GradNorm/dmd_train", generator_log_dict["dmdtrain_gradient_norm"].mean().item(), self.step)
                        else:
                            self.writer.add_scalar("Loss/critic", critic_log_dict["critic_loss"].mean().item(), self.step)
                            self.writer.add_scalar("GradNorm/critic", critic_log_dict["critic_grad_norm"].mean().item(), self.step)


                    if self.step % self.config.gc_interval == 0:
                        if dist.get_rank() == 0:
                            logging.info("DistGarbageCollector: Running GC.")
                        gc.collect()
                        torch.cuda.empty_cache()

                    if self.is_main_process:
                        current_time = time.time()
                        iteration_time = 0 if self.previous_time is None else current_time - self.previous_time
                        # MODIFIED: Log iteration time to TensorBoard
                        if self.writer:
                            self.writer.add_scalar("Time/per_iteration", iteration_time, self.step)
                        self.previous_time = current_time
                        # Log training progress
                        if TRAIN_GENERATOR and generator_log_dict:
                            print(f"step {self.step}, per iteration time {iteration_time}, generator_loss {generator_log_dict['generator_loss'].mean().item()}, generator_grad_norm {generator_log_dict['generator_grad_norm'].mean().item()}, dmdtrain_gradient_norm {generator_log_dict['dmdtrain_gradient_norm'].mean().item()}")
                        else:
                            print(f"step {self.step}, per iteration time {iteration_time}, critic_loss {critic_log_dict['critic_loss'].mean().item()}, critic_grad_norm {critic_log_dict['critic_grad_norm'].mean().item()}")

                    if self.step > self.config.max_iters:
                        break

        except Exception as e:
            if self.is_main_process:
                print(f"[ERROR] Training crashed at step {self.step} with exception: {e}")
                print(f"[ERROR] Exception traceback:", flush=True)
                import traceback
                traceback.print_exc()
        finally:
            # MODIFIED: Close the SummaryWriter
            if self.is_main_process and self.writer:
                self.writer.close()
