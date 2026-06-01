import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from typing import Dict, Any, Literal, Tuple
import os
import re
import time
from pathlib import Path
import mediapy

import numpy as np

import torch
import pytorch_lightning as pl
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)

from openai import OpenAI
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

from src.eval.models.base_model import BaseModel

# constants for video processing, taken from 
# https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

# Older (2.5-VL) version of the cookbood: https://github.com/QwenLM/Qwen3-VL/blob/e994ad452d36cbfaabccde0cb713c58d9d0d1c0e/cookbooks/video_understanding.ipynb
# latest version of qwen_vl_utils introduced a check that
# VIDEO_MIN_PIXELS should be <= VIDEO_MAX_PIXELS
# hence setting VIDEO_MIN_PIXELS to 32*28*28
# VIDEO_MAX_PIXELS is set to 48*28*28 to respect the 20480 token limit that we have now set
# taking inspiration from the cookbook: https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/video_understanding.ipynb
VIDEO_MIN_PIXELS = 16 * 28 * 28
VIDEO_MAX_PIXELS = 48 * 28 * 28 # number of pixels in each frame. changed this to 48 from 768 to respect the 20480 token limit that we have now set
FRAME_FACTOR = 2
FPS = 1.0 # changed this from 2.0 in the original code to 1.0 as our videos are sampled at 1 FPS
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768

# Set the maximum number of video token inputs.
# Here, 20480 represents the maximum number of input tokens for the VLLM model.
# Remember to adjust it according to your own configuration.
# NOTE: the original repository uses 128k instead of 24k but the notebook mentions that
# after 24k the performance will start to degrade. Hence sticking with 20480.
VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 20480 * 28 * 28)))


class Qwen2_5_VlHF(BaseModel):
    """
    Qwen2.5-VL model wrapper for Hugging Face Transformers.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", 
        change_sys_prompt: bool = False,
        forward_pipeline: Literal["qwen", "hf", "vllm_offline", "vllm_online"] = "qwen",
        add_vision_id: bool = True,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 1,
        do_sample: bool = False,
        dtype: str = "auto",
        max_new_tokens: int = 65536,
        device_map: Literal["auto", "cpu", "cuda"] = "auto",
        force_cpu: bool = False,
        vllm_gpu_util: float = 0.8,
        vllm_online_port: int = 8000,
        vllm_chat_template: str = Path(__file__).resolve().parent / "model_utils" / "qwen25_vl_vllm_online.jinja",
        **kwargs
    ):
        """Qwen2.5-VL model wrapper for Hugging Face Transformers.

        Args:
            model_name (str, optional): Model string. Defaults to "Qwen/Qwen2.5-VL-7B-Instruct".
            change_sys_prompt (bool, optional): Whether to change the system prompt. Defaults to False.
            forward_pipeline (Literal["qwen", "hf", "vllm_offline"], optional): Video processing backend. 
                - "qwen" uses the original repository example shown in: 
                - "hf" uses the autoprocessor for processing the video. 
                - "vllm_offline" uses the VLLM offline processing.
                - "vllm_online" uses the VLLM online processing.
                Defaults to "qwen".
            add_vision_id (bool, optional): Whether to add vision ID to the input prompts. Defaults to True.
            temperature (float, optional): Temperature for sampling. Defaults to 0.0.
            top_p (float, optional): Nucleus sampling probability. Defaults to 1.0.
            top_k (int, optional): Top-k sampling value. Defaults to 1.
            do_sample (bool, optional): Whether to use sampling. Defaults to True.
            dtype (str, optional): Data type for model loading. Defaults to "auto".
            vllm_gpu_util (float, optional): GPU memory utilization for VLLM. Defaults to 0.8.
            vllm_online_port (int, optional): Port for VLLM online server. Defaults to 8000.
            vllm_chat_template (str, optional): Chat template for VLLM online. Defaults to "qwen25_vl_vllm_online.jinja".
        kwargs: Additional keyword arguments for the base model.
            video_max_tokens (int, optional): Maximum number of video tokens in each frame. Defaults to 768.
            video_total_tokens (int, optional): Total number of video tokens across all frames. Defaults to 24000 * 0.9.
            fps (float, optional): Frames per second for video processing. Defaults to 1.0.
            fps_max_frames (int, optional): Maximum frames for video processing. Defaults to 768.
            max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 65536.
            device_map (Literal["auto", "cpu", "cuda"], optional): Device map for model loading. Defaults to "auto".
            force_cpu (bool, optional): Whether to force the model to run on CPU. Defaults to False.
        """
        super().__init__(**kwargs)
        torch_dtype = dtype
        
        self.forward_pipeline = forward_pipeline
        # import ipdb; ipdb.set_trace()
        if "vllm" not in self.forward_pipeline:
            if "2.5" in model_name:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_name,
                    device_map=device_map,
                    dtype=torch_dtype,
                )
                # import ipdb; ipdb.set_trace()

            else:
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    # torch_dtype=dtype,
                    device_map=device_map,
                    dtype=torch_dtype,
                )
            if force_cpu:
                self.model.to("cpu")
            self.model.eval()
        elif self.forward_pipeline == "vllm_online":
            self.model = OpenAI(base_url=f"http://localhost:{vllm_online_port}/v1", api_key="EMPTY")
        else:
            self.model = LLM(model=model_name, gpu_memory_utilization=vllm_gpu_util,
                             tensor_parallel_size=torch.cuda.device_count(),
                                    enforce_eager=True, limit_mm_per_prompt={"video": 1, "image": 2})

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            use_fast=True,
        )
        self.change_sys_prompt = change_sys_prompt
        self.model_name = model_name
        self.add_vision_id = add_vision_id
        
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        
        # For spatio-temporal granularity in video processing
        self.video_max_tokens: int = kwargs.get("video_max_tokens", 
                                                VIDEO_MAX_PIXELS // IMAGE_FACTOR // IMAGE_FACTOR)
        self.video_min_tokens: int = kwargs.get("video_min_tokens", VIDEO_MIN_PIXELS // IMAGE_FACTOR // IMAGE_FACTOR)
        self.video_total_tokens: int = kwargs.get("video_total_tokens", VIDEO_TOTAL_PIXELS // \
            IMAGE_FACTOR // IMAGE_FACTOR // FRAME_FACTOR)
        self._video_max_pixels = self.video_max_tokens * IMAGE_FACTOR * IMAGE_FACTOR
        self._video_min_pixels = self.video_min_tokens * IMAGE_FACTOR * IMAGE_FACTOR

        self.fps: float = kwargs.get("fps", FPS)
        self.fps_max_frames: int = kwargs.get("fps_max_frames", FPS_MAX_FRAMES) # this is similar to nframes for huggingface
        # total number of pixels in each frame
        self._video_total_pixels = self.video_total_tokens * IMAGE_FACTOR * IMAGE_FACTOR * FRAME_FACTOR
        self.sampling_params = None
        
        if self.forward_pipeline == "vllm_offline":
            self.sampling_params = SamplingParams(temperature=self.temperature, max_tokens=self.max_new_tokens)

        self.vllm_chat_template = vllm_chat_template
        self.vllm_chat_template_kwargs = kwargs.get("vllm_chat_template_kwargs", None)
        if self.vllm_chat_template:
            assert os.path.exists(self.vllm_chat_template), \
                f"VLLM chat template file {self.vllm_chat_template} does not exist."
            with open(self.vllm_chat_template, "r") as f:
                self.vllm_chat_template = f.read()
            if self.vllm_chat_template_kwargs is None:
                self.vllm_chat_template_kwargs = {"add_vision_id": self.add_vision_id}
            else:
                self.vllm_chat_template_kwargs["add_vision_id"] = self.add_vision_id

        
        
    def _adjust_st_granularity(
        self,
        orig_vid_res: Tuple[int, int],
        video_total_tokens: int,
        video_max_tokens: int = None,
        fps_max_frames: int = None,
    ):
        """provide one of video_max_tokens or fps_max_frames to adjust the spatio-temporal
        granularity of the video processing.
        
        Adding this in case we want to do ablation of total number of tokens Vs. max. number of frames Vs. 
        max. spatial resolution.
        
        Args:
            orig_vid_res (Tuple[int, int]): Original video resolution as a tuple (height, width).
            video_total_tokens (int): Total number of video tokens across all frames.
            video_max_tokens (int, optional): Maximum number of video tokens in each frame. Defaults to None.
            fps_max_frames (int, optional): Maximum frames for video processing. Defaults to None.
        Raises:
            AssertionError: If both video_max_tokens and fps_max_frames are provided.
            AssertionError: If neither video_max_tokens nor fps_max_frames is provided.
        Returns:
            None
        """
        
        assert not (video_total_tokens is not None and fps_max_frames is not None), "Only accept either video_max_tokens or fps_max_frames, not both."
        assert not (video_total_tokens is None and fps_max_frames is None), "Either video_max_tokens or fps_max_frames must be provided."
        
        small_dim_scale = min(orig_vid_res) / max(orig_vid_res)
        
        if video_max_tokens is not None:
            self.video_max_tokens = video_max_tokens
            self.video_total_tokens = video_total_tokens
            assert self._video_max_tokens <= self._video_total_tokens, \
                f"video_max_tokens {self._video_max_tokens} should be less than or equal to video_total_tokens {self._video_total_tokens}."
            
            self._video_max_pixels = self.video_max_tokens * IMAGE_FACTOR * IMAGE_FACTOR
            self._video_total_pixels = self.video_total_tokens * IMAGE_FACTOR * IMAGE_FACTOR * FRAME_FACTOR 

            self.fps_max_frames = (video_total_tokens * FRAME_FACTOR) // (video_max_tokens * video_max_tokens * small_dim_scale)
            assert self.fps_max_frames * self.video_max_tokens  <= self.video_total_tokens, \
                f"fps_max_frames {self.fps_max_frames} * video_max_tokens {self.video_max_tokens} should be less than or equal to video_total_tokens {self.video_total_tokens}."
                
        elif fps_max_frames is not None:
            self.fps_max_frames = fps_max_frames
            self.video_total_tokens = video_total_tokens
            
            self.video_max_tokens = round(np.sqrt(
                self.video_total_tokens * small_dim_scale * FRAME_FACTOR / self.fps_max_frames
            ))
            
            self._video_max_pixels = self.video_max_tokens * IMAGE_FACTOR * IMAGE_FACTOR
            self._video_total_pixels = self.video_total_tokens * IMAGE_FACTOR * IMAGE_FACTOR * FRAME_FACTOR
            assert self.fps_max_frames * self.video_max_tokens  <= self.video_total_tokens, \
                f"fps_max_frames {self.fps_max_frames} * video_max_tokens {self.video_max_tokens} should be less than or equal to video_total_tokens {self.video_total_tokens}."

    def subsampled_video_path(self, video_path, frame_idxs, video_tag):
        parent_dir = os.path.dirname(video_path)
        new_video_path = os.path.join(parent_dir, os.path.basename(video_path) + '_subsampled_qwen.mp4')

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

    def _qwen_video_proc_pipeline(self, messages: Dict[str, Any], max_new_tokens: int = None,
                                  verbose: bool=False) -> Dict[str, Any]:
        """
        Process video using the Qwen video processing pipeline.
        
        Args:
            messages (Dict[str, Any]): Messages containing video information.
            max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to None.
            verbose (bool, optional): Whether to print verbose output. Defaults to False
        
        Returns:
            Dict: Processed video information.
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        
        video_tag = None
        for msg in messages:
            if "content" in msg:
                if isinstance(msg["content"], list):
                    for content in msg["content"]:
                        if isinstance(content, dict) and content["type"] == "video":
                            content["max_pixels"] = self._video_max_pixels
                            content["min_pixels"] = self._video_min_pixels
                            content["total_pixels"] = self._video_total_pixels
                            content["fps"] = self.fps
                            content["fps_max_frames"] = self.fps_max_frames
                            video_tag = content["tag"]
                else:
                    if isinstance(msg["content"], dict) and msg["content"]["type"] == "video":
                        msg["content"]["max_pixels"] = self._video_max_pixels
                        msg["content"]["min_pixels"] = self._video_min_pixels
                        msg["content"]["total_pixels"] = self._video_total_pixels
                        msg["content"]["fps"] = self.fps
                        msg["content"]["fps_max_frames"] = self.fps_max_frames
                            
        # import ipdb; ipdb.set_trace()
        # import ipdb; ipdb.set_trace()
        image_inputs, video_inputs, video_kwargs = process_vision_info([messages], 
                                                                       image_patch_size=14, 
                                                                       return_video_kwargs=True,
                                                                   return_video_metadata=True)
        video_metadatas = None
        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None     

        # Do subsampling here for concat videos
        if video_tag is not None and 'concat' in video_tag:
            old_metadata = video_metadatas
            messages = self.concat_subsampling(messages, video_metadatas)
            image_inputs, video_inputs, video_kwargs = process_vision_info([messages], 
                                                                        image_patch_size=14, 
                                                                        return_video_kwargs=True,
                                                                    return_video_metadata=True)
            video_metadatas = None
            if video_inputs is not None:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
            else:
                video_metadatas = None 
            
            new_metadata = video_metadatas

            print("\nqwen subsampling: ", len(old_metadata[0]['frames_indices']))
            print("\ncustom + qwen subsampling: ", len(new_metadata[0]['frames_indices']))
            

        text = self.processor.apply_chat_template(messages, tokenize=False, 
                                                  add_generation_prompt=True, 
                                                  add_vision_id=self.add_vision_id)
        
        # fps_inputs = video_metadatas['fps']
        if verbose:
            if video_inputs is not None:
                print("video input:", video_inputs[0].shape)
        
        if verbose and video_inputs is not None:
            num_frames, _, resized_height, resized_width = video_inputs[0].shape
            print("num of video tokens:", int(num_frames / FRAME_FACTOR * resized_height / IMAGE_FACTOR * resized_width / IMAGE_FACTOR))
        
        
        if self.forward_pipeline != "vllm_offline":
            
            inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            
            generate_config = {
                "do_sample": self.do_sample,
            }
            if self.do_sample:
                generate_config["temperature"] = self.temperature
                generate_config["top_p"] = self.top_p
                generate_config["top_k"] = self.top_k
            else:
                # greedy decoding
                generate_config["num_beams"] = 1

            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **generate_config)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            return output_text[0]
        else:
            
            llm_inputs = {
                "prompt": text,
                "multi_modal_data": {
                    "image": image_inputs,
                    "video": video_inputs
                }
            }
            if video_inputs is None:
                llm_inputs["multi_modal_data"].pop("video")
            if image_inputs is None or len(image_inputs) == 0:
                llm_inputs["multi_modal_data"].pop("image")
            # import ipdb; ipdb.set_trace()
            outputs = self.model.generate([llm_inputs], sampling_params=self.sampling_params)
            return outputs[0].outputs[0].text 
    
        
    def _hf_video_proc_pipeline(self, messages: Dict[str, Any], max_new_tokens: int = None,
                                verbose: bool=False) -> Dict[str, Any]:
        """
        https://cornell-rgb.slack.com/archives/C094822705C/p1752521999559859
        https://github.com/huggingface/transformers/blob/6017f5e8ed33d48096cdf8630d1cc7cbf2550c90/src/transformers/models/qwen2_vl/video_processing_qwen2_vl.py#L167-L177
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        raise NotImplementedError("HF video processing pipeline is not implemented yet.")

    def create_prompt(
        self, 
        conversation: Dict[str, Any],
        **kwargs
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
                input_msg = {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": conversation[msg_idx]["content"]
                    }
                }
                
            elif conversation[msg_idx]["type"] == "image":
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

    def _qwen_vllm_online_pipeline(self, messages: Dict[str, Any], max_new_tokens: int = None,
                                  verbose: bool=False) -> Dict[str, Any]:
        """
        Process video using the Qwen VLLM online processing pipeline.
        
        Args:
            messages (Dict[str, Any]): Messages containing video information.
            max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to None.
            verbose (bool, optional): Whether to print verbose output. Defaults to False
        
        Returns:
            Dict: Processed video information.
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        
        for msg in messages:
            if "content" in msg:
                if isinstance(msg["content"], list):
                    for content in msg["content"]:
                        if isinstance(content, dict) and content["type"] == "image":
                            content["type"] = "image_url"
                            content["image_url"] = {
                                "url": f"file://{content['image']}"
                            }
                            content.pop("image")
                        elif isinstance(content, dict) and content["type"] == "video":
                            raise NotImplementedError("VLLM online video processing is not implemented yet.")
                else:
                    if isinstance(msg["content"], dict) and msg["content"]["type"] == "image":
                        msg["content"]["type"] = "image_url"
                        msg["content"]["image_url"] = {
                            "url": f"file://{msg['content']['image']}"
                        }
                        msg["content"].pop("image")
                    elif isinstance(msg["content"], dict) and msg["content"]["type"] == "video":
                        raise NotImplementedError("VLLM online video processing is not implemented yet.")

        extra_body = {}
        if self.vllm_chat_template is not None:
            extra_body["chat_template"] = self.vllm_chat_template
            extra_body["chat_template_kwargs"] = self.vllm_chat_template_kwargs
        extra_body["add_generation_prompt"] = True # will always be true since this is what we do in HF as well as per docs

        # import ipdb; ipdb.set_trace()
        response = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=self.temperature,
            extra_body=extra_body,
        )
        return response.choices[0].message.content    
    
    def forward(
        self,
        conversation: Dict[str, Any],
        max_new_tokens: int = 2048,
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
        # import ipdb; ipdb.set_trace()
        if self.forward_pipeline in ["qwen", "vllm_offline"]:
            return self._qwen_video_proc_pipeline(messages, max_new_tokens, verbose)
        elif self.forward_pipeline == "vllm_online":
            return self._qwen_vllm_online_pipeline(messages, max_new_tokens, verbose)
        elif self.forward_pipeline == "hf":
            return self._hf_video_proc_pipeline(messages, max_new_tokens, verbose)
        else:
            raise ValueError(f"Unknown forward pipeline: {self.forward_pipeline}")
    
    def post_process_response(self, response):
        return post_process_response(response)

def post_process_response(response: str):
    # pattern = re.compile(
    #     r'(?:(^[A-Z]$)|\{\s*\"answer\"\s*:\s*\"([A-Z])\"\s*\})'
    # )
    pattern = re.compile(
        r'(?:`*json\s*)*\{\s*(?:\"explanation\"\s*:\s*\".*?\"\s*,\s*)?\"answer\"\s*:\s*[\"\']*([A-Z])[\"\'\.]*|^[\"\']*([A-Z])[\"\'\.]*',
        re.MULTILINE
    )
    stripped_resp = response.strip()
    match = pattern.match(stripped_resp)
    # import ipdb; ipdb.set_trace()
    if match:
        return match.group(1) or match.group(2)
    else:
        return ""


def main(
    conv_fn: str,
    model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct",
    change_sys_prompt: bool = False,
    forward_pipeline: Literal["qwen", "hf"] = "qwen",
    add_vision_id: bool = True,
    video_max_tokens: int = VIDEO_MAX_PIXELS // IMAGE_FACTOR // IMAGE_FACTOR,
    video_total_tokens: int = VIDEO_TOTAL_PIXELS // IMAGE_FACTOR // IMAGE_FACTOR // FRAME_FACTOR,
    fps: float = FPS,
    fps_max_frames: int = FPS_MAX_FRAMES,
    device_map: Literal["auto", "cpu", "cuda", "null"] = "auto",
    force_cpu: bool = False,
    **kwargs: Any
):
    import yaml
    
    model = Qwen2_5_VlHF(
        model_name=model_name,
        change_sys_prompt=change_sys_prompt,
        forward_pipeline=forward_pipeline,
        add_vision_id=add_vision_id,
        video_max_tokens=video_max_tokens,
        video_total_tokens=video_total_tokens,
        fps=fps,
        fps_max_frames=fps_max_frames,
        device_map=device_map if device_map != "null" else None,
        force_cpu=force_cpu,
        temperature=0.0,
        do_sample=False,
    )
    
    with open(conv_fn, "r") as f:
        conversation = yaml.safe_load(f)
    start_time = time.time()
    with torch.inference_mode():
        output = model.forward(conversation, verbose=True)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print("Output:", output)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Qwen2.5-VL Model Inference")
    parser.add_argument("--conv_fn", type=str, default=str(root / "src/eval/models/dummy_data/conversation.yaml"), help="Path to the conversation YAML file.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", 
                        help="Model name to use.")
    parser.add_argument("--change_sys_prompt", action="store_true",
                        help="Whether to change the system prompt.")
    parser.add_argument("--forward_pipeline", type=str, choices=["qwen", "hf"],
                        default="qwen", help="Video processing pipeline to use.")
    parser.add_argument("--add_vision_id", action="store_true",
                        help="Whether to add vision ID to the input prompts.")
    parser.add_argument("--video_max_tokens", type=int, default=VIDEO_MAX_PIXELS // IMAGE_FACTOR // IMAGE_FACTOR,
                        help="Maximum number of video tokens in each frame.")
    parser.add_argument("--video_total_tokens", type=int, default=VIDEO_TOTAL_PIXELS // IMAGE_FACTOR // IMAGE_FACTOR // FRAME_FACTOR,
                        help="Total number of video tokens across all frames.")
    parser.add_argument("--fps", type=float, default=FPS,
                        help="Frames per second for video processing.")
    parser.add_argument("--fps_max_frames", type=int, default=FPS_MAX_FRAMES,
                        help="Maximum frames for video processing.")
    parser.add_argument("--device_map", type=str, choices=["auto", "cpu", "cuda", "null"],
                        default="auto", help="Device map for model loading.")
    parser.add_argument("--force_cpu", action="store_true",
                        help="Whether to force the model to run on CPU.")
    args = parser.parse_args()
    
    main(
        conv_fn=args.conv_fn,
        model_name=args.model_name,
        change_sys_prompt=args.change_sys_prompt,
        forward_pipeline=args.forward_pipeline,
        add_vision_id=args.add_vision_id,
        video_max_tokens=args.video_max_tokens,
        video_total_tokens=args.video_total_tokens,
        fps=args.fps,
        fps_max_frames=args.fps_max_frames,
        device_map=args.device_map if args.device_map != "null" else None,
        force_cpu=args.force_cpu
    )