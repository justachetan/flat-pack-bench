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
import torch
import pytorch_lightning as pl
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
)
from qwen_vl_utils import process_vision_info

from src.eval.models.base_model import BaseModel

class LlavaOV1_5(BaseModel):
    def __init__(
        self,
        model_name: Literal[
            "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct",
            "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        ] = "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct",
        change_sys_prompt: bool = False,
        temperature: float = 0.0,
        do_sample: bool = False,
        dtype: str = "bfloat16",
        vid_spl_mode: Literal["tokens", "frames"] = "frames",
        max_new_tokens: int = 1024,
        **kwargs: Any,
    ):
        """LlaVA-OneVision-1.5 model wrapper.

        Args:
            model_name (Literal[ &quot;llava, optional): model version name. Defaults to "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct".
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

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.vid_spl_mode = vid_spl_mode
 
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
            image_inputs, video_inputs = process_vision_info(messages)
        # import ipdb; ipdb.set_trace()
        elif self.vid_spl_mode == "tokens":
            raise NotImplementedError(
                "Token-based video sampling is not implemented yet. "
                "Please use 'frames' mode for now."
            )
        else:
            raise ValueError(f"Unknown video sampling mode: {self.vid_spl_mode}. ")

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device).to(self.model.dtype)

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
        }
        if not self.do_sample:
            generation_kwargs.pop("temperature", None)
        inputs.pop('second_per_grid_ts')
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
    model_name: str = "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct",
    change_sys_prompt: bool = False,
    **kwargs: Any
):
    import time
    import yaml
    
    model = LlavaOV1_5(
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
    parser.add_argument("--model_name", type=str, default="lmms-lab/LLaVA-OneVision-1.5-4B-Instruct", 
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