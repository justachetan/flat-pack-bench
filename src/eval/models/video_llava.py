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
    VideoLlavaProcessor,
    VideoLlavaForConditionalGeneration,
)
from loguru import logger as eval_logger

from src.eval.models.base_model import BaseModel, read_video_pyav
from src.eval.models.model_utils.video_subspl.subspl_concat_video import subspl_concat_video

def get_video_first_chat_template():
    """
    Returns a chat template that renders video content first, followed by images and text.
    This is to make the prompt similar to qwen-style models.
    """
    template_dict = {
        "chat_template": "{% for message in messages %}{{(message['role'] + ': ').upper()}}{# Render all videos first #}{% for content in message['content'] | selectattr('type', 'equalto', 'video') %}{{ '<video>' }}{% endfor %}{# Render all images next #}{% for content in message['content'] | selectattr('type', 'equalto', 'image') %}{{ '<image>' }}{% endfor %}{# Render all text next #}{% if message['role'] != 'assistant' %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{{ '\n' + content['text'] }}{% endfor %}{% else %}{% for content in message['content'] | selectattr('type', 'equalto', 'text') %}{% generation %}{{ '\n' + content['text'] }}{% endgeneration %}{% endfor %}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'ASSISTANT:\n' }}{% endif %}"
    }

    
    return template_dict["chat_template"]

"""
Source: https://huggingface.co/docs/transformers/en/model_doc/internvl#interleaved-image-and-video-inputs
Also checked that the processor output maintains the positioning of the media objects.
"""

class VideoLLaVA(BaseModel):
    def __init__(
        self,
        model_name: Literal[
            "LanguageBind/Video-LLaVA-7B-hf",
            "LanguageBind/Video-LLaVA-7B",
        ] = "LanguageBind/Video-LLaVA-7B-hf",
        change_sys_prompt: bool = False,
        temperature: float = 0.0,
        do_sample: bool = False,
        dtype: str = None,
        attn_implementation: str = None,
        **kwargs: Any,
    ):
        """Llava NeXT Video model wrapper.

        Args:
            model_name (Literal[ &quot;llava, optional): model version name. Defaults to "llava-hf/LLaVA-NeXT-Video-7B-hf".
            change_sys_prompt (bool, optional): edit the system prompt. Defaults to False.
            temperature (float, optional): temperature for generation. Defaults to 0.0.
            do_sample (bool, optional): enable random generation. False means greedy decoding. Defaults to False.
            dtype (str, optional): data type for the model weights. Defaults to "bfloat16".
            attn_implementation (str, optional): attention implementation to use. Defaults to "flash_attention_2".
            **kwargs: Any additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.change_sys_prompt = change_sys_prompt
        self.temperature = temperature
        self.do_sample = do_sample
        self._dtype = dtype
        self.attn_implementation = attn_implementation

        init_kwargs = {
            "dtype": self._dtype,
            "attn_implementation": self.attn_implementation,
        }
        if self._dtype is None:
            init_kwargs.pop("dtype", None)
        if self.attn_implementation is None:
            init_kwargs.pop("attn_implementation", None)

        self.model = VideoLlavaForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            **init_kwargs
        )
        self.model.eval()
        
        self.processor = VideoLlavaProcessor.from_pretrained(
            model_name,
            use_fast=True,
        )
        # NOTE: these values have been computed and set for images with 
        #       resolution 480x640 (HxW). If llava-next-video internally
        #       resizes to square images then these values will not need to be
        #       changed, but if not, then these values will need to be changed
        # TODO: these are redundant in this case. Right now we are directly using HF's 
        #      processor, but if we use our own processor then we will need to change and use these values
        # self.max_tokens = 10250 * 0.8 # maximum tokens that the model can handle is 10250 (verified from dry-runs)
        # self.toks_per_img = 2340 # again, verified from dry-runs
        # self.toks_per_vid_frame = 144 # verified from dry-runs, this is the number of tokens per video frame
        self._num_frames = 8 # Models were trained with 8 frames      
    
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
                input_msg = {
                    "role": "user",
                    "content": {
                        "type": "video",
                        "tag": conversation[msg_idx]["tag"],
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
    
    def load_images(self, messages):
        images = []
        for message in messages:
            if message["role"] != "user":
                continue

            for content in message["content"]:
                if content["type"] == "image":
                    images.append(Image.open(content["image"]))
        return images
    
    def load_video(self, video_path, max_num_frames=None, video_tag='video'):
        if max_num_frames is None:
            max_num_frames = self._num_frames

        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        total_frames = self.get_num_frames_in_video(video_path)
        if "concat" in video_tag:
            if "tracking" in video_tag:
                indices = np.concatenate((
                    np.array([0, 1]),
                    np.arange(2, total_frames, (total_frames - 2) / (max_num_frames - 2)).astype(int)
                ))
            else:
                indices = np.concatenate((
                    np.array([0]),
                    np.arange(1, total_frames, (total_frames - 1) / (max_num_frames - 1)).astype(int)
                ))
        else:
            indices = np.arange(0, total_frames, total_frames / max_num_frames).astype(int)
        print(indices)
        video = read_video_pyav(container, indices)
        return video
    
    def load_videos(self, messages, max_num_frames=None):
        videos = []
        for message in messages:
            if message["role"] != "user":
                continue

            for content in message["content"]:
                if content["type"] == "video":
                    videos.append(self.load_video(content["video"], max_num_frames, content["tag"]))
        return videos

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
            max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 60.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
        Returns:
            Dict[str, Any]: Model output.
        """

        messages = self.create_prompt(conversation, **kwargs)
        text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template=get_video_first_chat_template() 
        )
        media = {"images": self.load_images(messages), "videos": self.load_videos(messages)}
            
        if verbose:
            print(f"Messages:\n{json.dumps(messages, indent=2)}\n")
            print(f"Input text:\n{text}")
        
        if not media["images"]:
            media.pop("images")
        if not media["videos"]:
            media.pop("videos")
        inputs = self.processor(
                    text=text,
                    return_tensors="pt",
                    **media
        )

        inputs = inputs.to(self.model.device).to(self.model.dtype)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
        }
        if not self.do_sample:
            generation_kwargs.pop("temperature", None)
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
        r"^(?:[`]*json\s*)?\{\s*\"answer\"\s*:\s*[\"\']*([A-Z])[\"\']*\s*\}*|^[\"\']*([A-Z])[\"\'\.]*$",
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
    model_name: str = "LanguageBind/Video-LLaVA-7B-hf",
    change_sys_prompt: bool = False,
    **kwargs: Any
):
    import yaml
    
    model = VideoLLaVA(
        model_name=model_name,
        change_sys_prompt=change_sys_prompt,
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
    """
    concat_video_tracking:
    concat_video:
    """
    import argparse
    parser = argparse.ArgumentParser(description="Video-LLaVA Model Inference")
    parser.add_argument("--conv_fn", type=str, default=str(root / "src/eval/models/dummy_data/conversation.yaml"), help="Path to the conversation YAML file.")
    parser.add_argument("--model_name", type=str, default="LanguageBind/Video-LLaVA-7B-hf", 
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