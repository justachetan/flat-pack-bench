import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from typing import Dict, Any, Literal, List
import os
import re
import json
import time
import torch

import numpy as np
import warnings
from tqdm import tqdm
import argparse
import random
import torch.nn.functional as F
import mediapy

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info

from src.eval.models.base_model import BaseModel

class ArrowRL(BaseModel):
    def __init__(
                    self,
        model_name: Literal[
            "sherryxzh/ArrowRL-Qwen2.5-VL-7B",
        ] = "sherryxzh/ArrowRL-Qwen2.5-VL-7B",
        change_sys_prompt: bool = False,
        temperature: float = 0.0,
        do_sample: bool = False,
        dtype: str = "bfloat16",
        max_new_tokens: int = 1024,
        **kwargs: Any,
    ):
        """ArrowRL model wrapper.

        Args:
            model_name (Literal[ &quot;llava, optional): model version name. Defaults to "sherryxzh/ArrowRL-Qwen2.5-VL-7B".
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

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, 
            torch_dtype=torch.bfloat16,   
            attn_implementation="sdpa",
            device_map="auto" 
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_name
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

    def process_video_tensor_by_mode(self, video_list, mode):
        if isinstance(video_list, list):
            video = video_list[0]
        else:
            video = video_list
        assert mode in [0, 1, 2, 3, 4]
        if mode == 1:
            video = video.flip(0)  # reverse along time axis
        elif mode == 2:
            video = torch.cat([video, torch.zeros_like(video[0])[None], torch.zeros_like(video[0])[None], video.flip(0)], dim=0) # forward video, black frames, reverse video
        elif mode == 3:
            video = torch.cat([video.flip(0), torch.zeros_like(video[0])[None], torch.zeros_like(video[0])[None], video], dim=0) # reverse video, black frames, forward video
        elif mode == 4:  # shuffle
            video = video[torch.randperm(len(video))]
        if isinstance(video_list, list):
            return [video]
        return video

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
    
    def subsampled_video_path(self, video_path, frame_idxs, video_tag):
        parent_dir = os.path.dirname(video_path)
        new_video_path = os.path.join(parent_dir, os.path.basename(video_path) + '_subsampled_arrowrl.mp4')

        if os.path.exists(new_video_path):
            return new_video_path
        
        frames = mediapy.read_video(video_path)
        # subsampled_frames = [frames[i] for i in frame_idxs]
        
        subsampled_frame_idxs = []
        # NOTE (ac): this should keep the first two frames and then evenly sample the rest
        if 'tracking' in video_tag:
            subsampled_frame_idxs = [0, 1] + np.linspace(2, len(frames)-1, num=len(frame_idxs)-2, dtype=int).tolist()
        else:
            subsampled_frame_idxs = [0] + np.linspace(1, len(frames)-1, num=len(frame_idxs)-1, dtype=int).tolist()
        subsampled_frames = [frames[i] for i in subsampled_frame_idxs]
        mediapy.write_video(new_video_path, subsampled_frames, fps=1)

        print("length of subsampled video")
        print(len(subsampled_frames))
        
        return new_video_path


    def concat_subsampling(self, messages, video_metadatas):
        video_idx = 0
        for message in messages:
            if 'content' not in message:
                continue

            for content in message['content']:
                if content['type'] != 'video' or 'concat' not in content['tag']:
                    continue
            
                frame_idxs = video_metadatas[video_idx]['frames_indices']
                original_video_path = content['video']
                new_video_path = self.subsampled_video_path(original_video_path, frame_idxs, content['tag'])
                print("\ncustom subsampling: ", len(mediapy.read_video(new_video_path)))
                content['video'] = new_video_path
                video_idx += 1
        return messages
    
    def forward(
        self,
        conversation: Dict[str, Any],
        max_new_tokens: int = None,
        verbose: bool = False,
        **kwargs
    ):
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        
        video_tag = None
        messages = self.create_prompt(conversation, **kwargs)
        for message in messages:
            if message["role"] != "user":
                continue

            for content in message["content"]:
                if content["type"] == "video":
                    content["fps"] = 1
                    video_tag = content["tag"]
        
        if verbose:
            print(f"Messages:\n{json.dumps(messages, indent=2)}\n")
            input_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )
            print(f"Input prompt:\n{input_prompt}\n")
        

        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
            return_video_metadata=True
        )

        video_metadatas = None
        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None     

        # Do subsampling here for concat videos
        if 'concat' in video_tag:
            old_metadata = video_metadatas
            messages = self.concat_subsampling(messages, video_metadatas)
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                 messages,
                 return_video_kwargs=True,
                 return_video_metadata=True
            )
                                                                        
            video_metadatas = None
            if video_inputs is not None:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
            else:
                video_metadatas = None 
            
            new_metadata = video_metadatas

            print("\nqwen subsampling: ", len(old_metadata[0]['frames_indices']))
            print("\ncustom + qwen subsampling: ", len(new_metadata[0]['frames_indices']))

        frame_process_mode = 0
        video_inputs = self.process_video_tensor_by_mode(video_inputs, frame_process_mode)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # start_process_time = time.time()
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        # end_process_time = time.time()
        # print(f"Preprocessing time: {end_process_time - start_process_time:.2f} seconds")

        inputs = inputs.to(self.model.device)
        inputs['pixel_values_videos'] = inputs['pixel_values_videos'].to(torch.bfloat16)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
            "use_cache": True
        }
        if not self.do_sample:
            generation_kwargs.pop("temperature", None)
        
        # start_generation_time = time.time()
        output_ids = self.model.generate(
            **inputs,
            **generation_kwargs
        )
        # end_generation_time = time.time()
        # print(f"Generation time: {end_generation_time - start_generation_time:.2f} seconds")
        
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        # start_decode_time = time.time()
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        # end_decode_time = time.time()
        # print(f"Decoding time: {end_decode_time - start_decode_time:.2f} seconds")
        
        return response[0]
    
    def post_process_response(self, response):
        return post_process_response(response)

def post_process_response(response: str):
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
    model_name: str = "sherryxzh/ArrowRL-Qwen2.5-VL-7B",
    change_sys_prompt: bool = False,
    **kwargs: Any
):
    import yaml
    
    model = ArrowRL(
        model_name=model_name,
        change_sys_prompt=change_sys_prompt,
        do_sample=False,
        temperature=0.0,
        **kwargs
    )
    # import ipdb; ipdb.set_trace()
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
    """
    concat_video_tracking:
    concat_video:
    """
    import argparse
    parser = argparse.ArgumentParser(description="ArrowRL Model Inference")
    parser.add_argument("--conv_fn", type=str, default=str(root / "src/eval/models/dummy_data/conversation.yaml"), help="Path to the conversation YAML file.")
    parser.add_argument("--model_name", type=str, default="sherryxzh/ArrowRL-Qwen2.5-VL-7B", 
                        help="Model name to use.")
    parser.add_argument("--change_sys_prompt", action="store_true",
                        help="Whether to change the system prompt.")
    
    args = parser.parse_args()
    
    main(
        conv_fn=args.conv_fn,
        model_name=args.model_name,
        change_sys_prompt=args.change_sys_prompt,
        max_new_tokens=256,
    )