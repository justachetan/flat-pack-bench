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
    AutoProcessor,
    AutoModelForImageTextToText,
)
from loguru import logger as eval_logger

from src.eval.models.base_model import BaseModel, read_video_pyav
from src.eval.models.model_utils.video_subspl.subspl_concat_video import subspl_concat_video

"""
Source: https://huggingface.co/docs/transformers/en/model_doc/internvl#interleaved-image-and-video-inputs
Also checked that the processor output maintains the positioning of the media objects.
"""

class InternVL3(BaseModel):
    def __init__(
        self,
        model_name: Literal[
            "OpenGVLab/InternVL3-14B-hf",
            "OpenGVLab/InternVL3-38B-hf",
            "OpenGVLab/InternVL3-78B-hf",
        ] = "OpenGVLab/InternVL3-38B-hf",
        change_sys_prompt: bool = False,
        temperature: float = 0.0,
        do_sample: bool = False,
        dtype: str = "bfloat16",
        vid_spl_mode: Literal["tokens", "frames"] = "frames",
        max_new_tokens: int = 1024,
        num_beams: int = 1,
        top_p: float = 1.0,
        top_k: int = 1,
        **kwargs: Any,
    ):
        """Llava NeXT Video model wrapper.

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
            top_p (float, optional): top-p sampling parameter. Defaults to 1.0.
            top_k (int, optional): top-k sampling parameter. Defaults to 1.
            num_beams (int, optional): number of beams for beam search. Defaults to 1.
            **kwargs: Any additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.change_sys_prompt = change_sys_prompt
        self.temperature = temperature
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.top_p = top_p
        self.top_k = top_k

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            use_fast=True,
        )
        self.vid_spl_mode = vid_spl_mode
        # NOTE: these values have been computed and set for images with 
        #       resolution 480x640 (HxW). If llava-next-video internally
        #       resizes to square images then these values will not need to be
        #       changed, but if not, then these values will need to be changed
        # TODO: these are redundant in this case. Right now we are directly using HF's 
        #      processor, but if we use our own processor then we will need to change and use these values
        self.max_tokens = 10250 * 0.8 # maximum tokens that the model can handle is 10250 (verified from dry-runs)
        self.toks_per_img = 2340 # again, verified from dry-runs
        self.toks_per_vid_frame = 144 # verified from dry-runs, this is the number of tokens per video frame
        self._num_frames = 32 # the InternVL3 paper and VSI-Bench have both used 32 frames as one of their settings so we will use that as well        

        self._tmp_video_dir = kwargs.get("tmp_video_dir", None)
    
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

            elif conversation[msg_idx]["type"] == "video":                    
                # continue
                video_path = conversation[msg_idx]["content"]
                if conversation[msg_idx]["tag"].startswith("concat"):
                    initial_preserve_indices = [0] # preserve the first frame as visual prompt for concat videos
                    if "tracking" in conversation[msg_idx]["tag"]:
                        # for tracking videos, we preserve the first two frames to provide better visual prompt for tracking
                        initial_preserve_indices = [0, 1]
                    output_video_subspl_dir = self._tmp_video_dir if self._tmp_video_dir is not None else None
                    video_path = subspl_concat_video(
                        video_path,
                        output_dir=output_video_subspl_dir, # this way it will be saved in the same directory as the input video, i.e., the media cache
                        num_subspl_frames=self._num_frames,
                        initial_preserve_indices=initial_preserve_indices,
                    )
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
            input_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )
            print(f"Input prompt:\n{input_prompt}\n")

        
        # import ipdb; ipdb.set_trace()
        inputs = None
        if self.vid_spl_mode == "frames":
            # import ipdb; ipdb.set_trace()
            # NOTE: in case of multiple videos, select the minimum number of frames
            #      across all videos, and then sample that many frames from each video
            num_frames_in_videos = [self.get_num_frames_in_video(
                    content["video"]
                ) for message in messages for content in message["content"] if content["type"] == "video"]
            min_num_frames = None
            if len(num_frames_in_videos) > 0:
                min_num_frames = min(num_frames_in_videos)
            
            # import ipdb; ipdb.set_trace()
            processor_kwargs = {
                "tokenize": True,
                "return_dict": True,
                "return_tensors": "pt",
                "padding": True,
                "add_generation_prompt": True,
            }
            if min_num_frames is not None:
                processor_kwargs["num_frames"] = min(self._num_frames, min_num_frames)
                processor_kwargs["video_load_backend"] = "decord"
            
            inputs = self.processor.apply_chat_template(
                messages,
                **processor_kwargs,
            )
        
        # import ipdb; ipdb.set_trace()
        elif self.vid_spl_mode == "tokens":
            raise NotImplementedError(
                "Token-based video sampling is not implemented yet. "
                "Please use 'frames' mode for now."
            )
        else:
            raise ValueError(f"Unknown video sampling mode: {self.vid_spl_mode}. ")

        inputs = inputs.to(self.model.device).to(self.model.dtype)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
            "num_beams": self.num_beams,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        if not self.do_sample:
            generation_kwargs.pop("temperature", None)
            generation_kwargs["num_beams"] = 1
            generation_kwargs.pop("top_p", None)
            generation_kwargs.pop("top_k", None)
        output_ids = self.model.generate(
            **inputs,
            **generation_kwargs
        ).to("cpu")
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        
        return response[0]
    
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
    model_name: str = "OpenGVLab/InternVL3-14B-hf",
    change_sys_prompt: bool = False,
    **kwargs: Any
):
    import time
    import yaml
    
    model = InternVL3(
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
    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3-14B-hf", 
                        help="Model name to use.")
    parser.add_argument("--change_sys_prompt", action="store_true",
                        help="Whether to change the system prompt.")
    
    args = parser.parse_args()
    
    main(
        conv_fn=args.conv_fn,
        model_name=args.model_name,
        change_sys_prompt=args.change_sys_prompt,
    )
    

    # 0a5c9ad55e7653e673cddb48d1d5cbf7d724c7509d8e2d9d7ecfc6ceab204f6c_SEP_MEDIA_FIRST_55f006ffeb0558b6a2b1b3f913bdb6ca49526e072c97c7ab8a9f0f4fb216129f