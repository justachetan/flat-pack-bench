import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from typing import Dict, Any, Literal, Tuple, List
import os
import re
import json
import tempfile

import av

import torch
import numpy as np
from PIL import Image
import pytorch_lightning as pl
from transformers import (
    AutoModel,
    AutoTokenizer,
)
from loguru import logger as eval_logger

from decord import VideoReader, cpu
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from src.eval.models.base_model import BaseModel, read_video_pyav

"""
Source: https://huggingface.co/docs/transformers/en/model_doc/internvl#interleaved-image-and-video-inputs
Also checked that the processor output maintains the positioning of the media objects.
"""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class InternVL3_5(BaseModel):
    def __init__(
        self,
        model_name: Literal[
            "OpenGVLab/InternVL3_5-1B-HF",
            "OpenGVLab/InternVL3_5-2B-HF",
            "OpenGVLab/InternVL3_5-4B-HF",
            "OpenGVLab/InternVL3_5-8B-HF",
            "OpenGVLab/InternVL3_5-14B-HF",
            "OpenGVLab/InternVL3_5-38B-HF",
            "OpenGVLab/InternVL3_5-20B-A4B-HF",
            "OpenGVLab/InternVL3_5-30B-A3B-HF",
            "OpenGVLab/InternVL3_5-241B-A28B-HF"
        ] = "OpenGVLab/InternVL3_5-1B-HF",
        change_sys_prompt: bool = False,
        temperature: float = 0.0,
        do_sample: bool = False,
        dtype: str = "bfloat16",
        vid_spl_mode: Literal["tokens", "frames"] = "frames",
        max_new_tokens: int = 1024,
        **kwargs: Any,
    ):
        """InternVL3.5 Video model wrapper.

        Args:
            model_name (Literal[ &quot;llava, optional): model version name. Defaults to "llava-hf/LLaVA-NeXT-Video-7B-hf".
            change_sys_prompt (bool, optional): edit the system prompt. Defaults to False.
            temperature (float, optional): temperature for generation. Defaults to 0.0.
            do_sample (bool, optional): enable random generation. False means greedy decoding. Defaults to False.
            dtype (str, optional): data type for the model weights. Defaults to "bfloat16".
            video_spl_mode (Literal[&quot;tokens&quot;, &quot;frames&quot;], optional): video sampling mode.
                "tokens" means that video is sampled with token constraints,
                "frames" means that video is sampled with frame constraints. 
                When using multiple videos in prompts, prefer "tokens". Defaults to "frames".
                TODO: outline strategies for sampling for multiple videos.
            max_new_tokens (int, optional): maximum number of tokens to generate. Defaults to 1024.
            **kwargs: Any additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.change_sys_prompt = change_sys_prompt
        self.temperature = temperature
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True,
        )
        self.vid_spl_mode = vid_spl_mode

        # According to example: https://huggingface.co/OpenGVLab/InternVL3_5-8B#inference-with-transformers
        self.num_frames = 32
        self.input_size = 448

    def build_transform(self, input_size):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def dynamic_preprocess(self, image, image_size=None):
        if image_size is None:
            image_size = self.input_size
        processed_images = []
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image_file, input_size=None):
        if input_size is None:
            input_size = self.input_size
        image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def get_index(self, bound, fps, max_frame, first_idx=0, num_segments=None):
        if num_segments is None:
            num_segments = self.num_frames
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(num_segments)
        ])
        return frame_indices

    def load_video(self, video_path, bound=None, input_size=None, num_segments=None):
        if input_size is None:
            input_size = self.input_size
        if num_segments is None:
            num_segments = self.num_frames
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        pixel_values_list, num_patches_list = [], []
        transform = self.build_transform(input_size=input_size)
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
            img = self.dynamic_preprocess(img, image_size=input_size)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    def create_prompt(
        self,
        conversation: List[Dict[str, Any]],
        **kwargs,
    ):
        
        messages = []
        for msg_idx in range(len(conversation)):
            
            input_msg = None
            
            if conversation[msg_idx]["tag"] == "task_instruction":
                # NOTE: this needs to be handled differently from the other
                # message types as it can potentially be a system prompt
                input_msg = {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": conversation[msg_idx]["content"]
                    }
                }
                if self.change_sys_prompt:
                    input_msg["role"] = "system"
                else:
                    input_msg["content"] = [input_msg["content"]]
                if len(messages) == 0 or messages[-1]["role"] != "user":
                    messages.append(input_msg)
                else:
                    messages[-1]["content"].extend(input_msg["content"])
                continue

            elif conversation[msg_idx]["tag"] == "video":
                # continue
                video_path = conversation[msg_idx]["content"]
                input_msg = {
                    "role": "user",
                    "content": {
                        "type": "video",
                        "video": video_path,
                    }
                }
                
            elif conversation[msg_idx]["type"] == "text":
                # continue
                input_msg = {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": conversation[msg_idx]["content"]
                    }
                }
                
            elif conversation[msg_idx]["type"] == "image":
                # continue
                image_path = conversation[msg_idx]["content"]
                input_msg = {
                    "role": "user",
                    "content": {
                        "type": "image",
                        "image": image_path,
                    }
                }
            else:
                raise ValueError(f"Unknown message type: {conversation[msg_idx]['type']}")
                
            input_msg["content"] = [input_msg["content"]]
            if len(messages) == 0 or messages[-1]["role"] != "user":
                messages.append(input_msg)
            else:
                messages[-1]["content"].extend(input_msg["content"])
                
        return messages
    
    def get_num_frames_in_video(
        self,
        video_path: str,
    ):
        """Uses pyav to get total number of frames in a video file.
        Args:
            video_path (str): The path to the video file.
        """
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        return total_frames
    
    def forward(
        self,
        conversation: Dict[str, Any],
        max_new_tokens: int = None,
        verbose: bool = False,
        
        **kwargs
    ):
        """Forward pass through the model.
        Args:
            conversation (Dict[str, Any]): Conversation messages.
            max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 2048.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
        Returns:
            Dict[str, Any]: Model output.
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        messages = self.create_prompt(conversation, **kwargs)
        if verbose:
            print(f"Messages:\n{json.dumps(messages, indent=2)}\n")

        
        # import ipdb; ipdb.set_trace()
        text = ""
        media = None
        num_patches_list = []
        if self.vid_spl_mode == "frames":
            for message in messages:
                for content in message["content"]:
                    if content["type"] == "image":
                        image = self.load_image(content["image"]).to(self.model.device).to(self.model.dtype)
                        if media is None:
                            media = image
                        else:
                            media = torch.cat((media, image), dim=0)
                        num_patches_list.append(image.size(0))
                        text += "Image: <image>"
                    elif content["type"] == "video":
                        video, video_num_patches_list = self.load_video(content["video"])
                        video = video.to(self.model.device).to(self.model.dtype)
                        if media is None:
                            media = video
                        else:
                            media = torch.cat((media, video), dim=0)
                        num_patches_list.extend(video_num_patches_list)
                        text += "".join([f"Video Frame {i + 1}: <image>" for i in range(len(video_num_patches_list))])
                    else:
                        text += content["text"]
                    
        
        # import ipdb; ipdb.set_trace()
        elif self.vid_spl_mode == "tokens":
            raise NotImplementedError(
                "Token-based video sampling is not implemented yet. "
                "Please use 'frames' mode for now."
            )
        else:
            raise ValueError(f"Unknown video sampling mode: {self.vid_spl_mode}. ")

        if verbose:
            print(f"Input prompt:\n{text}\n")

        generation_config = dict(max_new_tokens=self.max_new_tokens, do_sample=self.do_sample)
        response, history = self.model.chat(self.tokenizer,
                                            media,
                                            text,
                                            generation_config,
                                            num_patches_list=num_patches_list,
                                            history=None,
                                            return_history=True)
        return response
    
    def post_process_response(self, response):
        return post_process_response(response)

def post_process_response(response: str):
    # pattern = re.compile(
    #     r'(?:(^[A-Z]$)|\{\s*\"answer\"\s*:\s*\"([A-Z])\"\s*\})'
    # )
    pattern = re.compile(
        r'(?:`*json\s*)*\{\s*(?:\"explanation\"\s*:\s*\".*?\"\s*,\s*)?\"answer\"\s*:\s*[\"\']*([A-Z])[\"\'\.]*|^[\"\']*([A-Z])[\"\'\.]*',
        re.MULTILINE,
    )
    stripped_resp = response.strip()
    match = pattern.match(stripped_resp)
    if match:
        return match.group(1) or match.group(2)
    else:
        return ""
    
def main(
    conv_fn: str,
    model_name: str = "OpenGVLab/InternVL3_5-1B-HF",
    change_sys_prompt: bool = False,
    **kwargs: Any
):
    import time
    import yaml
    
    model = InternVL3_5(
        model_name=model_name,
        change_sys_prompt=change_sys_prompt,
        do_sample=False,
        temperature=0.0,
        **kwargs
    )
    
    with open(conv_fn, "r") as f:
        conversation = yaml.safe_load(f)
    start_time = time.time()
    output = model.forward(conversation, verbose=True)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print("Output:", output)
    output = model.post_process_response(output)
    print("Post-processed output:", output)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="InternVL3 Model Inference")
    parser.add_argument("--conv_fn", type=str, default=str(root / "src/eval/models/dummy_data/conversation.yaml"), help="Path to the conversation YAML file.")
    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3_5-8B", 
                        help="Model name to use.")
    parser.add_argument("--change_sys_prompt", action="store_true",
                        help="Whether to change the system prompt.")
    
    args = parser.parse_args()
    
    main(
        conv_fn=args.conv_fn,
        model_name=args.model_name,
        change_sys_prompt=args.change_sys_prompt,
    )
    

    # 