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
import time
import torch
import pytorch_lightning as pl
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import mediapy
import numpy as np

from src.eval.models.base_model import BaseModel


# Following the cookbook directly here
# taking inspiration from the cookbook: https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/video_understanding.ipynb

SAMPLE_FPS = 1.0 # changed this from 2.0 in the original code to 1.0 as our videos are sampled at 1 FPS
MIN_PIXELS = 64 * 32 * 32
MIN_FRAMES = 4
MAX_FRAMES = 2048

# Set the maximum number of video token inputs.
# Here, 20480 represents the maximum number of input tokens for the VLLM model.
# Remember to adjust it according to your own configuration.
# NOTE: the original repository uses 128k instead of 24k but the notebook mentions that
# after 24k the performance will start to degrade. Hence sticking with 20480.
VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 20480 * 32 * 32)))


class Qwen3_VlHF(BaseModel):
    """
    Qwen3-VL model wrapper for Hugging Face Transformers.
    """
    
    def __init__(
        self,
        model_name: Literal[
            "Qwen/Qwen3-VL-4B-Instruct",
            "Qwen/Qwen3-VL-8B-Instruct",
            "Qwen/Qwen3-VL-32B-Instruct"
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
            "Qwen/Qwen3-VL-235B-A22B-Instruct",
            "Qwen/Qwen3-VL-4B-Thinking",
            "Qwen/Qwen3-VL-8B-Thinking",
            "Qwen/Qwen3-VL-32B-Thinking",
            "Qwen/Qwen3-VL-30B-A3B-Thinking",
            "Qwen/Qwen3-VL-235B-A22B-Thinking",
        ] = "Qwen/Qwen3-VL-4B-Instruct", 
        change_sys_prompt: bool = False,
        forward_pipeline: Literal["qwen", "hf", "vllm_offline"] = "qwen",
        add_vision_id: bool = True,
        temperature: float = 0.0,
        do_sample: bool = False,
        dtype: str = "auto",
        max_new_tokens: int = 2048,
        device_map: Literal["auto", "cpu", "cuda"] = "auto",
        force_cpu: bool = False,
        vllm_gpu_util: float = 0.95,
        **kwargs
    ):
        """Qwen3-VL model wrapper for Hugging Face Transformers.

        Args:
            model_name (str, optional): Model string. Defaults to "Qwen/Qwen3-VL-4B-Instruct".
            change_sys_prompt (bool, optional): Whether to change the system prompt. Defaults to False.
            forward_pipeline (Literal["qwen", "hf", "vllm_offline"], optional): Video processing backend. 
                - "qwen" uses the original repository example shown in: 
                - "hf" uses the autoprocessor for processing the video. 
                - "vllm_offline" uses the VLLM offline processing.
                Defaults to "qwen".
            add_vision_id (bool, optional): Whether to add vision ID to the input prompts. Defaults to True.
            temperature (float, optional): Temperature for sampling. Defaults to 0.0.
            do_sample (bool, optional): Whether to use sampling. Defaults to True.
        kwargs: Additional keyword arguments for the base model.
            min_pixels (int, optional): Minimum number of pixels for video frames. Defaults to MIN_PIXELS.
            max_frames (int, optional): Maximum number of frames for video processing. Defaults to MAX_FRAMES.
            fps (float, optional): Frames per second for video processing. Defaults to SAMPLE_FPS.
            video_total_pixels (int, optional): Total number of pixels for video processing. Defaults to VIDEO_TOTAL_PIXELS.
            max_new_tokens (int, optional): Maximum number of new tokens to generate. Defaults to 2048.
            device_map (Literal["auto", "cpu", "cuda"], optional): Device map for model loading. Defaults to "auto".
            force_cpu (bool, optional): Whether to force the model to run on CPU. Defaults to False.
        """
        super().__init__(**kwargs)
        torch_dtype = dtype
        
        self.forward_pipeline = forward_pipeline
        # import ipdb; ipdb.set_trace()
        if self.forward_pipeline != "vllm_offline":
            if "-A" in model_name:
                # for MOE models
                self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                    model_name,
                    device_map=device_map,
                    dtype=torch_dtype,
                )
                # import ipdb; ipdb.set_trace()

            else:
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_name,
                    # torch_dtype=dtype,
                    device_map=device_map,
                    dtype=torch_dtype,
                )
            if force_cpu:
                self.model.to("cpu")
            self.model.eval()
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
        self.max_new_tokens = max_new_tokens
        
        self.min_pixels = kwargs.get("min_pixels", MIN_PIXELS)
        self.max_frames = kwargs.get("max_frames", MAX_FRAMES)
        self.sample_fps = kwargs.get("fps", SAMPLE_FPS)
        self.video_total_pixels = kwargs.get("video_total_pixels", VIDEO_TOTAL_PIXELS)

        self.sampling_params = None
        
        if self.forward_pipeline == "vllm_offline":
            os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
            self.sampling_params = SamplingParams(temperature=self.temperature, max_tokens=self.max_new_tokens)
    
    def subsampled_video_path(self, video_path, frame_idxs, video_tag):
        parent_dir = os.path.dirname(video_path)
        new_video_path = os.path.join(parent_dir, os.path.basename(video_path).split('.')[0] + '_subsampled_qwen_3_vl.mp4')

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
                            content["min_pixels"] = self.min_pixels
                            content["total_pixels"] = self.video_total_pixels
                            content["max_frames"] = self.max_frames
                            content["sample_fps"] = self.sample_fps
                            video_tag = content["tag"]
                else:
                    if isinstance(msg["content"], dict) and msg["content"]["type"] == "video":
                        content["min_pixels"] = self.min_pixels
                        content["total_pixels"] = self.video_total_pixels
                        content["max_frames"] = self.max_frames
                        content["sample_fps"] = self.sample_fps
                            
        # import ipdb; ipdb.set_trace()
        text = self.processor.apply_chat_template(messages, tokenize=False, 
                                                  add_generation_prompt=True, 
                                                  add_vision_id=self.add_vision_id)
        # import ipdb; ipdb.set_trace()
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages],
            image_patch_size=16, # has been updated to 16 from 14 for qwen3 vl
            return_video_kwargs=True,
            return_video_metadata=True)

        # fps_inputs = video_kwargs['fps']
        
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
            image_inputs, video_inputs, video_kwargs = process_vision_info([messages], 
                                                                        image_patch_size=16, 
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
        
        if verbose:
            if video_inputs is not None:
                print("video input:", video_inputs[0].shape)
            
        inputs = self.processor(text=[text], 
                                images=image_inputs, 
                                videos=video_inputs, 
                                # fps=fps_inputs,
                                video_metadata=video_metadatas,
                                **video_kwargs, 
                                do_resize=False, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        
        generate_config = {
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            generate_config["temperature"] = self.temperature

        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, **generate_config)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return output_text[0]
        
    
    def _vllm_offline_video_proc_pipeline(self, messages: List[Dict[str, Any]], max_new_tokens: int = None,
                                         verbose: bool=False) -> Dict[str, Any]:
        """
        Process video using the VLLM offline video processing pipeline.        
        Source: https://github.com/QwenLM/Qwen3-VL?tab=readme-ov-file#offline-inference
        """
        
        # NOTE: assumes single vide input for now, which is the case for our benchmark. 
        # Will need to be updated if we want to support multiple video inputs.
        video_tag = None
        for msg in messages:
            if "content" in msg:
                if isinstance(msg["content"], list):
                    for content in msg["content"]:
                        if isinstance(content, dict) and content["type"] == "video":
                            video_tag = content["tag"]
                
        text = self.processor.apply_chat_template(messages, tokenize=False, 
                                                add_generation_prompt=True, 
                                                add_vision_id=self.add_vision_id)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages],
            image_patch_size=16, # has been updates to 16 from 14 for qwen3 vl
            return_video_kwargs=True,
            return_video_metadata=True)

        # Do subsampling here for concat videos
        if 'concat' in video_tag:
            
            video_metadatas = None
            if video_inputs is not None:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
            else:
                video_metadatas = None       

            old_metadata = video_metadatas
            messages = self.concat_subsampling(messages, video_metadatas)
            image_inputs, video_inputs, video_kwargs = process_vision_info([messages], 
                                                                        image_patch_size=16, 
                                                                        return_video_kwargs=True,
                                                                    return_video_metadata=True)
            new_video_metadatas = None
            if video_inputs is not None:
                new_video_inputs, new_video_metadatas = zip(*video_inputs)
                new_video_inputs, new_video_metadatas = list(new_video_inputs), list(new_video_metadatas)
            else:
                new_video_metadatas = None 
            
            new_metadata = new_video_metadatas

            if verbose:
                print("\nqwen subsampling: ", len(old_metadata[0]['frames_indices']))
                print("\ncustom + qwen subsampling: ", len(new_metadata[0]['frames_indices'])) 
        
        if verbose:
            if video_inputs is not None:
                print("video input:", video_inputs[0].shape)

        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs

        llm_inputs = {
            "prompt": text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs
        }
        
        inputs = [llm_inputs]
        outputs = self.model.generate(inputs, sampling_params=self.sampling_params)
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
        max_new_tokens = self.max_new_tokens if hasattr(self, "max_new_tokens") else max_new_tokens
        messages = self.create_prompt(conversation, **kwargs)
        # import ipdb; ipdb.set_trace()
        if self.forward_pipeline == "qwen":
            return self._qwen_video_proc_pipeline(messages, max_new_tokens, verbose)
        elif self.forward_pipeline == "hf":
            return self._hf_video_proc_pipeline(messages, max_new_tokens, verbose)
        elif self.forward_pipeline == "vllm_offline":
            return self._vllm_offline_video_proc_pipeline(messages, max_new_tokens, verbose)
        else:
            raise ValueError(f"Unknown forward pipeline: {self.forward_pipeline}")
    
    def post_process_response(self, response):
        return post_process_response(response)

def post_process_response(response: str, debug: bool = False) -> str:
    # pattern = re.compile(
    #     r'(?:(^[A-Z]$)|\{\s*\"answer\"\s*:\s*\"([A-Z])\"\s*\})'
    # )
    
    think_tag = "</think>"
    index = response.find(think_tag)
    if index != -1:
        response = response[index + len(think_tag):]
        if debug:
            print(response[:100])
    else:
        print(response)
    
    pattern = re.compile(
        r'(?:["\'`]*json\s*)*\{\s*(?:\"explanation\"\s*:\s*\".*?\"\s*,\s*)?\"answer\"\s*:\s*[\"\']*([A-Z])[\"\'\.]*|^[\"\']*([A-Z])[\"\'\.]*',
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
    model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
    change_sys_prompt: bool = False,
    forward_pipeline: Literal["qwen", "hf"] = "qwen",
    add_vision_id: bool = True,
    device_map: Literal["auto", "cpu", "cuda", "null"] = "auto",
    force_cpu: bool = False,
    **kwargs: Any
):
    import yaml
    
    model = Qwen3_VlHF(
        model_name=model_name,
        change_sys_prompt=change_sys_prompt,
        forward_pipeline=forward_pipeline,
        add_vision_id=add_vision_id,
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
    """
    concat_video_tracking:
    concat_video:
    """
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3-VL Model Inference")
    parser.add_argument("--conv_fn", type=str, default=str(root / "src/eval/models/dummy_data/conversation.yaml"), help="Path to the conversation YAML file.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-4B-Instruct", 
                        help="Model name to use.")
    parser.add_argument("--change_sys_prompt", action="store_true",
                        help="Whether to change the system prompt.")
    parser.add_argument("--forward_pipeline", type=str, choices=["qwen", "hf"],
                        default="qwen", help="Video processing pipeline to use.")
    parser.add_argument("--add_vision_id", action="store_true",
                        help="Whether to add vision ID to the input prompts.")
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
        device_map=args.device_map if args.device_map != "null" else None,
        force_cpu=args.force_cpu
    )