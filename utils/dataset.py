# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
from torch.utils.data import Dataset
import torch
import json
import datasets
import pandas as pd

class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class TwoTextDataset(Dataset):
    """Dataset that returns two text prompts per sample for prompt-switch training.

    The dataset behaves similarly to :class:`TextDataset` but instead of a single
    prompt, it provides *two* prompts – typically the first prompt is used for the
    first segment of the video, and the second prompt is used after a temporal
    switch during training.

    Args:
        prompt_path (str): Path to a text file containing the *first* prompt for
            each sample. One prompt per line.
        switch_prompt_path (str): Path to a text file containing the *second*
            prompt for each sample. Must have the **same number of lines** as
            ``prompt_path`` so that prompts are paired 1-to-1.
    """
    def __init__(self, prompt_path: str, switch_prompt_path: str):
        # Load the first-segment prompts.
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        # Load the second-segment prompts.
        with open(switch_prompt_path, encoding="utf-8") as f:
            self.switch_prompt_list = [line.rstrip() for line in f]

        assert len(self.switch_prompt_list) == len(self.prompt_list), (
            "The two prompt files must contain the same number of lines so that "
            "each first-segment prompt is paired with exactly one second-segment prompt."
        )

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        return {
            "prompts": self.prompt_list[idx],            # first-segment prompt
            "switch_prompts": self.switch_prompt_list[idx],  # second-segment prompt
            "idx": idx,
        }


class MultiTextDataset(Dataset):
    """Dataset for multi-segment prompts stored in a JSONL file.

    Each line is a JSON object, e.g.
        {"prompts": ["a cat", "a dog", "a bird"]}

    Args
    ----
    prompt_path : str
        Path to the JSONL file
    field       : str
        Name of the list-of-strings field, default "prompts"
    cache_dir   : str | None
        ``cache_dir`` passed to HF Datasets (optional)
    """

    def __init__(self, prompt_path: str, field: str = "prompts", cache_dir: str | None = None):
        self.ds = datasets.load_dataset(
            "json",
            data_files=prompt_path,
            split="train",
            cache_dir=cache_dir,
            streaming=False, 
        )

        assert len(self.ds) > 0, "JSONL is empty"
        assert field in self.ds.column_names, f"Missing field '{field}'"

        seg_len = len(self.ds[0][field])
        for i, ex in enumerate(self.ds):
            val = ex[field]
            assert isinstance(val, list), f"Line {i} field '{field}' is not a list"
            assert len(val) == seg_len,  f"Line {i} list length mismatch"

        self.field = field

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        return {
            "idx": idx,
            "prompts_list": self.ds[idx][self.field],  # List[str]
        }


def cycle(dl):
    while True:
        for data in dl:
            yield data


import ast
import decord

class MultiShots_FrameConcat_Dataset(Dataset):
    def __init__(self, csv_path):
        import ast

        metadata = pd.read_csv(csv_path)
        # shot_num_from_caption,json_path,video_path,frame_number
        self.video_path = metadata["video_path"].to_list() if "video_path" in metadata.columns else None
        self.caption_json_path = metadata["json_path"].to_list() if "json_path" in metadata.columns else None
        self.frame_number = metadata['frame_number'].to_list() if 'frame_number' in metadata.columns else None

    def __len__(self):
        return len(self.caption_json_path)

    def __getitem__(self, idx):
        while True:
            try:
                if self.frame_number is not None:
                    frame_number = ast.literal_eval(self.frame_number[idx]) # shot information
                    max_frame_number = frame_number[-1][-1]
                    shot_flag = []
                    for shot_index in range(len(frame_number)):
                        shot_flag += [shot_index] * (frame_number[shot_index][1] - frame_number[shot_index][0])
                    shot_flag = shot_flag + shot_flag[:-1]

                if self.video_path is not None:
                    video_path = self.video_path[idx]  # read video
                    frames, frame_indexes = self.read_video(video_path, max_frame_number)
                    frames = 2.0 * frames - 1.0  # normalization
                    shot_flag = [shot_flag[i] for i in frame_indexes]

                global_captions, shots_captions = [], []  # read caption
                caption_json_path = self.caption_json_path[idx]
                with open(caption_json_path, 'r') as f:
                    caption_content = json.load(f)
                    global_captions.append(caption_content["global_caption"])
                    shots_caption = []
                    for i in range(len(frame_number)):
                        # shots_caption.append([f'shot{i}:' + caption_content[f'shot{i}']])
                        shots_caption.append([f'shot{i}:' + caption_content[f'shot{i+1}']])  # [NOTE] data index begin with Shot 1
                    shots_captions.append(shots_caption) 
                refers = caption_content.get("refers", [])
                if self.video_path is not None:
                    batch = {
                        "data_path": video_path if self.video_path is not None else None,
                        "data": frames if self.video_path is not None else None,
                        "global_captions": global_captions,
                        "shots_captions": shots_captions,
                        "refers": refers,
                        "shot_flag": shot_flag if self.frame_number is not None else None,
                        "idx": idx,
                    }
                else:
                    batch = {
                        "global_captions": global_captions,
                        "shots_captions": shots_captions,
                        "refers": refers,
                        "shot_flag": shot_flag,
                        "idx": idx,
                    }

                return batch
            
            except Exception as e:
                idx = (idx + 1) % len(self.video_path)
                print(f"[ERROR] Load data index {idx} occurs error {e}.")

    def read_video(self, video_path, max_frame_number):
        ctx = decord.cpu(0)
        reader = decord.VideoReader(video_path, ctx=ctx, height=480, width=832) # [Hard Code], Only for Wan training
        length = len(reader)
        length = min(length, max_frame_number)
        frame_indexes = range(length)
        frame_indexes = [min(frame_index, length - 1) for frame_index in frame_indexes]
        if len(frame_indexes) % 4 != 1:
            frame_indexes = frame_indexes[ :(len(frame_indexes)-1) // 4 * 4 + 1]
        frames = reader.get_batch(frame_indexes)
        frames = frames.asnumpy() / 255.0
        return frames, frame_indexes

    def custom_collate_fn(self, examples):
        data_paths = [example["data_path"] for example in examples]
        data  = [example["data"] for example in examples]
        global_captions  = [example["global_captions"] for example in examples]
        shots_captions  = [example["shots_captions"] for example in examples]
        shot_flag  = [example["shot_flag"] for example in examples]
        refers = [example.get("refers", []) for example in examples]
        idx  = [example["idx"] for example in examples]
        return {
                "data_path": data_paths,
                "data": data,
                "global_captions": global_captions,
                "shots_captions": shots_captions,
                "refers": refers,
                "shot_flag": shot_flag,
                "idx": idx,
            }
    

class ODE_Sample_Dataset(Dataset):
    def __init__(self, csv_path):

        metadata = pd.read_csv(csv_path)
        # shot_num_from_caption,json_path,video_path,frame_number
        self.dict_path = metadata["latent_path"].to_list()

    def __len__(self):
        return len(self.dict_path)

    def __getitem__(self, idx):
        while True:
            # try:
            dict_path = self.dict_path[idx]
            data_all = torch.load(dict_path, map_location='cpu')

            noise = data_all['noise']
            caption = data_all['caption']
            condition_latents = data_all['condition_latents']
            latent_0_input = data_all['0_input']  # [1, f, c, h, w]
            latent_13_input = data_all['13_input']
            latent_25_input = data_all['25_input']
            latent_37_input = data_all['37_input']
            pred_x0_input = data_all['pred_x0']
            latent_all = torch.stack([latent_0_input, latent_13_input, latent_25_input, latent_37_input, pred_x0_input], dim=1)
            shot_flags_for_rope = data_all['shot_flags_for_rope'] if 'shot_flags_for_rope' in data_all.keys() else None

            batch = {
                "dict_path": dict_path,
                "noise": noise,
                "caption": caption,
                "condition_latents": condition_latents,
                "latent_0_input": latent_0_input,
                "latent_13_input": latent_13_input,
                "latent_25_input": latent_25_input,
                "latent_37_input": latent_37_input,
                "pred_x0_input": pred_x0_input,
                "latent_all": latent_all,
                "shot_flags_for_rope": shot_flags_for_rope,
                "idx": idx,
            }

            return batch

    def custom_collate_fn(self, examples):

        dict_path = [example["dict_path"] for example in examples]
        noise  = [example["noise"] for example in examples]
        caption  = [example["caption"] for example in examples]
        condition_latents  = [example["condition_latents"] for example in examples]
        latent_0_input  = [example["latent_0_input"] for example in examples]
        latent_13_input  = [example["latent_13_input"] for example in examples]
        latent_25_input  = [example["latent_25_input"] for example in examples]
        latent_37_input  = [example["latent_37_input"] for example in examples]
        pred_x0_input  = [example["pred_x0_input"] for example in examples]
        latent_all = [example["latent_all"] for example in examples]
        shot_flags_for_rope  = [example["shot_flags_for_rope"] for example in examples]
        idx  = [example["idx"] for example in examples]
        return {
                "dict_path": dict_path,
                "noise": noise,
                "caption": caption,
                "condition_latents": condition_latents,
                "latent_0_input": latent_0_input,
                "latent_13_input": latent_13_input,
                "latent_25_input": latent_25_input,
                "latent_37_input": latent_37_input,
                "pred_x0_input": pred_x0_input,
                "latent_all": latent_all,
                "shot_flags_for_rope": shot_flags_for_rope,
                "idx": idx,
            }
