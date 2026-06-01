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
    LlavaOnevisionForConditionalGeneration,
)
from loguru import logger as eval_logger

from src.eval.models.base_model import BaseModel, read_video_pyav
from src.eval.models.model_utils.video_subspl.subspl_concat_video import subspl_concat_video

def get_video_first_chat_template():
    """
    Returns a chat template that renders video content first, followed by images and text.
    This is to make the prompt similar to qwen-style models.
    
    Implemented after confirming that default ordering HF is 
    arbitrary, and there is no official recommendation for 
    video-first rendering.
    """
    template_dict = {
        "chat_template": "{% for message in messages %}{{'<|im_start|>' + message['role'] + ' '}}{# Render all videos first #}{% for content in message['content'] | selectattr('type', 'equalto', 'video') %}{{ '<video>' }}{% endfor %}{# Render all images next #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ '\n' + content['text'] }}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ '\n' + content['text'] }}{% endgeneration %}{% endfor %}{% endif %}{{'<|im_end|>'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    }

    
    return template_dict["chat_template"]


class LlavaOV(BaseModel):
    def __init__(
        self,
        model_name: Literal[
            "llava-hf/llava-onevision-qwen2-7b-ov-hf",
            "llava-hf/llava-onevision-qwen2-72b-ov-hf",
        ] = "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        change_sys_prompt: bool = False,
        temperature: float = 0.0,
        do_sample: bool = False,
        dtype: str = "bfloat16",
        mixed_media_mode: Literal["default", "video_first"] = "video_first",
        vid_spl_mode: Literal["frames", "tokens"] = "frames",
        **kwargs: Any,
    ):
        """Llava NeXT Video model wrapper.

        Args:
            model_name (Literal[ &quot;llava, optional): model version name. Defaults to "llava-hf/LLaVA-NeXT-Video-7B-hf".
            change_sys_prompt (bool, optional): edit the system prompt. Defaults to False.
            temperature (float, optional): temperature for generation. Defaults to 0.0.
            do_sample (bool, optional): enable random generation. False means greedy decoding. Defaults to False.
            dtype (str, optional): data type for the model weights. Defaults to "bfloat16".
            mixed_media_mode (Literal[&quot;default&quot;, &quot;video_first&quot;], optional): video placed first in the rendering order before images. Defaults to "video_first".
                "video_first" is used to make the prompt similar to qwen-style models. "default" is the default rendering order of the model with the 
                images first, then videos, and finally text.
            video_spl_mode (Literal[&quot;tokens&quot;, &quot;frames&quot;], optional): video sampling mode.
                "tokens" means that video is sampled with token constraints,
                "frames" means that video is sampled with frame constraints. 
                When using multiple videos in prompts, prefer "tokens". Defaults to "frames".
                TODO: outline strategies for sampling for multiple videos.
            **kwargs: Any additional keyword arguments.
            # TODO: add a token-based video frame sampling mode apart from the default _n_frame (number of frames)
                    based sampling strategy.
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.change_sys_prompt = change_sys_prompt
        self.temperature = temperature
        self.do_sample = do_sample
        self.mixed_media_mode = mixed_media_mode
        
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
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
        self._num_frames = 32 # paper says that model was trained at most on 32 frames per video, so we will use that as the default
        self._tmp_video_subspl_dir = kwargs.get("tmp_video_dir", None)
        eval_logger.info(f"Model {self.model_name} initialized with _n_frames={self._num_frames}.")
        
        
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
                    output_video_subspl_dir = self._tmp_video_subspl_dir if self._tmp_video_subspl_dir is not None else None
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
    
    def load_video(
        self,
        video_path: str,
        max_num_frames: int = None,
    ):
        """Uses pyav to load a video file similar to how it is shown here
        https://huggingface.co/docs/transformers/main/en/model_doc/llava-next-video
        
        Only `self.max_num_frames` frames are loaded from the video.

        Args:
            video_path (str): The path to the video file.
        """
        if max_num_frames is None:
            max_num_frames = self._num_frames
            
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        max_num_frames = min(max_num_frames, total_frames)
        indices = np.arange(0, total_frames, total_frames / max_num_frames).astype(int)
        clip = read_video_pyav(container, indices)
        return clip
    
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
    
    def load_image(self,
                   image_path: str,):
        """Loads an image from the given path.
        Args:
            image_path (str): The path to the image file.
        Returns:
            PIL.Image.Image: The loaded image.
        """
        return Image.open(image_path)
        
    
    def forward(
        self,
        conversation: Dict[str, Any],
        max_new_tokens: int = 60,
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
        
        messages = self.create_prompt(conversation, **kwargs)
        if verbose:
            print(f"Messages:\n{json.dumps(messages, indent=2)}\n")
            input_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                chat_template=get_video_first_chat_template() if self.mixed_media_mode == "video_first" else None,
            )
            print(f"Input prompt:\n{input_prompt}\n")

        
        inputs = None
        if self.vid_spl_mode == "frames":
            # import ipdb; ipdb.set_trace()
            # NOTE: in case of multiple videos, select the minimum number of frames
            #      across all videos, and then sample that many frames from each video
            min_num_frames = min([
                self.get_num_frames_in_video(
                    content["video"]
                ) for message in messages for content in message["content"] if content["type"] == "video"
            ])
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                num_frames=min(self._num_frames, min_num_frames),
                add_generation_prompt=True,
                video_load_backend="decord",
                chat_template=get_video_first_chat_template() if self.mixed_media_mode == "video_first" else None,
            )
        
        # import ipdb; ipdb.set_trace()
        elif self.vid_spl_mode == "tokens":
            raise NotImplementedError(
                "Token-based video sampling is not implemented yet. "
                "Please use 'frames' mode for now."
            )
        else:
            raise ValueError(f"Unknown video sampling mode: {self.vid_spl_mode}. ")

        inputs = inputs.to(self.model.device)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
        }
        if not self.do_sample:
            generation_kwargs.pop("temperature", None)
            generation_kwargs["num_beams"] = 1
        output_ids = self.model.generate(
            **inputs,
            **generation_kwargs,
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
    model_name: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    change_sys_prompt: bool = False,
    mixed_media_mode: Literal["default", "video_first"] = "video_first",
    **kwargs: Any
):
    import yaml
    
    model = LlavaOV(
        model_name=model_name,
        change_sys_prompt=change_sys_prompt,
        mixed_media_mode=mixed_media_mode,
        do_sample=False,
        temperature=0.0,
        **kwargs
    )
    
    with open(conv_fn, "r") as f:
        conversation = yaml.safe_load(f)
    
    output = model.forward(conversation, verbose=True)
    print("Output:", output)
    output = model.post_process_response(output)
    print("Post-processed output:", output)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLaVA-OneVision Model Inference")
    parser.add_argument("--conv_fn", type=str, default=str(root / "src/eval/models/dummy_data/conversation.yaml"), help="Path to the conversation YAML file.")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-onevision-qwen2-7b-ov-hf", 
                        help="Model name to use.")
    parser.add_argument("--change_sys_prompt", action="store_true",
                        help="Whether to change the system prompt.")
    parser.add_argument("--mixed_media_mode", type=str, default="video_first",
                        choices=["default", "video_first"],
                        help="Rendering order for mixed media. 'video_first' places videos before images.")
    parser.add_argument("--tmp_video_dir", type=str, default=None,
                        help="Temporary directory to save subsampled videos. If not set, subsampled videos will be saved in the same directory as the input videos.")

    args = parser.parse_args()
    
    main(
        conv_fn=args.conv_fn,
        model_name=args.model_name,
        change_sys_prompt=args.change_sys_prompt,
        mixed_media_mode=args.mixed_media_mode,
        tmp_video_dir=args.tmp_video_dir,
    )
    

    # 0a5c9ad55e7653e673cddb48d1d5cbf7d724c7509d8e2d9d7ecfc6ceab204f6c_SEP_MEDIA_FIRST_55f006ffeb0558b6a2b1b3f913bdb6ca49526e072c97c7ab8a9f0f4fb216129f