import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from typing import Dict, Any, Literal, List
import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import copy
from decord import VideoReader, cpu
import numpy as np
import json
import re

from src.eval.models.base_model import BaseModel

class LlavaVideo(BaseModel):
    def __init__(
            self,
            model_name: Literal[
                "lmms-lab/LLaVA-Video-7B-Qwen2",
                "lmms-lab/LLaVA-Video-72B-Qwen2"
            ] = "lmms-lab/LLaVA-Video-7B-Qwen2",
            change_sys_prompt: bool = False,
            temperature: float = 0.0,
            do_sample: bool = False,
            dtype: str = "bfloat16",
            max_new_tokens: int = 4096,
            max_frames_num: int = 64,
            **kwargs: Any,
    ):
        """LLaVA-Video model wrapper.

        Args:
            model_name (Literal[ &quot;llava, optional): model version name. Defaults to "lmms-lab/LLaVA-Video-7B-Qwen2".
            change_sys_prompt (bool, optional): edit the system prompt. Defaults to False.
            temperature (float, optional): temperature for generation. Defaults to 0.0.
            do_sample (bool, optional): enable random generation. False means greedy decoding. Defaults to False.
            dtype (str, optional): data type for the model weights. Defaults to "bfloat16".
            video_spl_mode (Literal[&quot;tokens&quot;, &quot;frames&quot;], optional): video sampling mode.
                "tokens" means that video is sampled with token constraints,
                "frames" means that video is sampled with frame constraints. 
                When using multiple videos in prompts, prefer "tokens". Defaults to "frames".
                TODO: outline strategies for sampling for multiple videos.
            max_new_tokens (int, optional): maximum number of tokens to generate. Defaults to 4096.
            max_frames_num (int, optional): maximum number of video frames to load. Defaults to 64
            **kwargs: Any additional keyword arguments.
        """

        super().__init__(**kwargs)
        self.model_name = model_name
        self.change_sys_prompt = change_sys_prompt
        self.temperature = temperature
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.max_frames_num = max_frames_num

        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            self.model_name,
            None,
            "llava_qwen",
            torch_dtype=dtype,
            attn_implementation="sdpa",
            device_map="auto"
        )
        self.model.eval()

    def load_video(self, video_path, max_frames_num, fps=1, force_sample=False, video_tag="video"):
        if max_frames_num == 0:
            return np.zeros((1, 336, 336, 3))
        vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps()/fps)
        frame_idx = [i for i in range(0, len(vr), fps)]
        frame_time = [i/fps for i in frame_idx]
        if len(frame_idx) > max_frames_num or force_sample:
            sample_fps = max_frames_num
            if "concat" in video_tag:
                if "tracking" in video_tag:
                    uniform_sampled_frames = [0, 1] + np.linspace(2, total_frame_num - 1, sample_fps - 2, dtype=int).tolist()
                else:
                    uniform_sampled_frames = [0] + np.linspace(1, total_frame_num - 1, sample_fps - 1, dtype=int).tolist()
            else:
                uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int).tolist()
            frame_idx = uniform_sampled_frames
            frame_time = [i/vr.get_avg_fps() for i in frame_idx]
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        return spare_frames, frame_time, video_time
    
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
            max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 4096.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
        Returns:
            Dict[str, Any]: Model output.
        """
         
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        messages = self.create_prompt(conversation, **kwargs)

        if verbose:
            print(f"Messages:\n{json.dumps(messages, indent=2)}\n")
        
        text = ""
        video_path = None
        video_tag = None
        for message in messages:
            for content in message["content"]:
                if content["type"] == "video":
                    video_path = content["video"]
                    video_tag = content["tag"]
                elif content["type"] == "text":
                    text += content["text"]

        video, frame_time, video_time = self.load_video(video_path, self.max_frames_num, 1, force_sample=True, video_tag=video_tag)
        video = self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().bfloat16()
        video = [video]
        conv_template = "qwen_1_5"
        time_instruction = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."

        question = DEFAULT_IMAGE_TOKEN + f"{time_instruction}\n{text}"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        input_ids = tokenizer_image_token(
            prompt_question,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)
        generate_config = {}
        if self.do_sample:
            generate_config["temperature"] = self.temperature
        else:
            generate_config["num_beams"] = 1
        cont = self.model.generate(
            input_ids,
            images=video,
            modalities= ["video"],
            # temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            **generate_config
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
        return text_outputs
    
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
    model_name: str = "lmms-lab/LLaVA-Video-7B-Qwen2",
    change_sys_prompt: bool = False,
    **kwargs: Any
):
    import time
    import yaml
    
    model = LlavaVideo(
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
    """
    concat_video_tracking:
    concat_video:
    """
    import argparse
    parser = argparse.ArgumentParser(description="LLaVA-Video Model Inference")
    parser.add_argument("--conv_fn", type=str, default=str(root / "src/eval/models/dummy_data/conversation.yaml"), help="Path to the conversation YAML file.")
    parser.add_argument("--model_name", type=str, default="lmms-lab/LLaVA-Video-7B-Qwen2", 
                        help="Model name to use.")
    parser.add_argument("--change_sys_prompt", action="store_true",
                        help="Whether to change the system prompt.")
    
    args = parser.parse_args()
    
    main(
        conv_fn=args.conv_fn,
        model_name=args.model_name,
        change_sys_prompt=args.change_sys_prompt,
    )