import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import os
import os.path as osp
import json
import time
import base64
from copy import deepcopy
from io import BytesIO
from typing import List, Union, Literal, Dict, Any

import mediapy
import imageio.v2 as iio
from PIL import Image

import numpy as np

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

from src.eval.models.base_model import BaseModel
from src.benchmark.lmmeval.models.utils import retry_with_exponential_backoff
from src.eval.prompts.templates.questions.common import calculate_sha256
from src.eval.models.qwen_2_5_vl_hf import post_process_response as qwen_post_process_response

# as per limits shared here: https://platform.openai.com/docs/guides/images-vision?api-mode=chat
MAX_FRAMES = 500
API_TYPE = os.getenv("API_TYPE", "openai")
NUM_SECONDS_TO_SLEEP = 30
from loguru import logger as eval_logger

if API_TYPE == "openai":
    API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
    API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
elif API_TYPE == "azure":
    API_URL = os.getenv("AZURE_ENDPOINT", "https://api.cognitive.microsoft.com/sts/v1.0/issueToken")
    API_KEY = os.getenv("AZURE_API_KEY", "YOUR_API_KEY")
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json",
    }

MAX_TRIES= int(os.getenv("OPENAI_MAX_TRIES_INT")) if os.getenv("OPENAI_MAX_TRIES_INT") is not None else 10


MODEL_VERSIONS = [
    "gpt-4.1-2025-04-14",
    "o4-mini-2025-04-16",
    "gpt-4o-2024-08-06",
    "gpt-5-2025-08-07"
]

class OpenAIGPT(BaseModel):
    
    def __init__(
        self,
        model_name: Literal["gpt-4.1-2025-04-14", "o4-mini-2025-04-16", "gpt-4o-2024-08-06", "gpt-5-2025-08-07"] = "gpt-5-2025-08-07",
        api_key: str = API_KEY,
        force_media_marks_in_prompt: bool = True,
        video_mark: str = "Here is the video showing the assembly process: ",
        visual_prompt_mark: str = "Here is the visual prompt: ",
        alternative_visual_prompt_marks: List[str] = ["Here is Image A: ", "Here is Image B: "],
        temperature: float = 0.0,
        seed: int = 42,
        **kwargs
    ):
        
        super().__init__(**kwargs)
        self.model_version = model_name
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key)

        if not os.path.isdir('gpt_cache'):
            os.makedirs('gpt_cache')
        self.force_media_marks_in_prompt = force_media_marks_in_prompt
        self.video_mark = video_mark
        self.visual_prompt_mark = visual_prompt_mark
        self.alternative_visual_prompt_marks = alternative_visual_prompt_marks
        
        self.temperature = temperature
        if self.model_version.startswith("o") or self.model_version.startswith("gpt-5"):
            eval_logger.warning(
                f"Model {self.model_version} does not support temperature setting. Reverting to default (1.0)."
            )
            self.temperature = 1.0
        self.seed = seed
        self._debug = kwargs.get("debug", False)
        self._debug_subspl_video_dir = kwargs.get("debug_subspl_video_dir", osp.join(root, "tmp"))

    def _load_img(self, img: Union[Image.Image, str], output_format: Literal["JPEG", "PNG"] = "JPEG") -> str:
        """
        Load an image and return it as a base64-encoded string.
        """
        if isinstance(img, str):
            img = iio.imread(img)
            img = Image.fromarray(img)
        
        buffer = BytesIO()
        img.save(buffer, format=output_format)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
        
    def _load_video(
            self, 
            video_path: Union[str, List[str]], 
            num_frame_limit: int=500, 
            output_format: Literal["JPEG", "PNG"] = "JPEG",
            preserve_initial_indices: List[int] = [],
        ) -> str:
        """
        Load a video and return it as a base64-encoded string.
        
        Currently only takes path to .mp4 files.

        preserve_initial_indices: List[int]: list of frame indices at the beginning
            to preserve in the sampled frames. Useful for concat videos where the 
            initial frames contain visual prompts.

        TODO: make this generic enough to handle directory of frames of list 
        of frame images
        """
        
        video_frames = []
        video = mediapy.read_video(video_path)
        for frame in video:
            img = Image.fromarray(frame)
            img_b64 = self._load_img(img, output_format=output_format)
            video_frames.append(img_b64)
        
        if len(video_frames) > num_frame_limit:
            eval_logger.warning(
                f"Video {video_path} has more than {num_frame_limit} frames. "
                f"Sampling {num_frame_limit} frames uniformly from the video."
            )
            frame_idxs = preserve_initial_indices + np.linspace(len(preserve_initial_indices), 
                                                                len(video_frames) - 1, 
                                                                num_frame_limit - len(preserve_initial_indices)
                                                    ).astype(int).tolist()
            # frame_idxs = np.linspace(0, len(video_frames) - 1, num_frame_limit).astype(int)
            if self._debug:
                eval_logger.debug(f"Sampling frames at indices: {frame_idxs}")
                subspl_video = video[np.array(frame_idxs).astype(int)]
                output_fn = f"{osp.basename(video_path).split('.')[0]}_{'-'.join(map(str, preserve_initial_indices))+'_'}num_frames_{num_frame_limit}.mp4"
                mediapy.write_video(osp.join(self._debug_subspl_video_dir, output_fn), subspl_video, fps=1)
            video_frames = [video_frames[idx] for idx in frame_idxs]

        return video_frames
        
    def create_prompt(self, conversation: List[Dict[str, Any]], **kwargs):
        """
        Create a prompt from the conversation data.
        """
        
        messages = []
        
        for msg_idx in range(len(conversation)):
            
            if conversation[msg_idx]["tag"] == "task_instruction":
                input_msg = {
                    "role": "system",
                    "content": conversation[msg_idx]["content"]
                }
                messages = [input_msg] + messages
                
            elif conversation[msg_idx]["type"] == "video":
                
                if self.force_media_marks_in_prompt:
                    # for OpenAI, we need to mark the video frames and the visual prompts
                    # because it does not support video input natively
                    mark_container = {
                        "type": "text",
                        "text": self.video_mark
                    }
                    if messages and messages[-1]["role"] == "user":
                        if "content" in messages[-1]:
                            messages[-1]["content"].append(mark_container)
                        else:
                            messages[-1]["content"] = [mark_container]   
                    else:
                        messages.append(
                            {
                                "role": "user",
                                "content": [mark_container]
                            }
                        )
                    
                
                input_msg = list()
                num_vps = 0
                for tmp in conversation:
                    if tmp["type"] == "image":
                        num_vps += 1
                preserve_initial_indices = []
                if conversation[msg_idx]["tag"].startswith("concat"):
                    assert num_vps == 0, "Currently we do not support concat videos with visual prompts. \
                        Please preprocess the video to include the visual prompts in the initial frames."
                    preserve_initial_indices = [0]
                    if "tracking" in conversation[msg_idx]["tag"]:
                        preserve_initial_indices = [0, 1]

                video_frames_b64 = self._load_video(
                    conversation[msg_idx]["content"],
                    num_frame_limit=MAX_FRAMES-num_vps,
                    preserve_initial_indices=preserve_initial_indices,
                )
                
                for frame in video_frames_b64:
                    frame_container = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
                    }
                    if messages and messages[-1]["role"] == "user":
                        if "content" in messages[-1]:
                            messages[-1]["content"].append(frame_container)
                        else:
                            messages[-1]["content"] = [frame_container]
                    else:
                        messages.append(
                            {
                                "role": "user",
                                "content": [frame_container]
                            }
                        )
            
            elif conversation[msg_idx]["type"] == "image":
                vp_mark = self.visual_prompt_mark
                if self.force_media_marks_in_prompt:
                    # for OpenAI, we need to mark the video frames and the visual prompts
                    # because it does not support video input natively
                    
                    # check if the visual prompt is jumbled or not
                    # if it is jumbled, we need to use the alternative visual prompt marks
                    for tmp in conversation:
                        if tmp["tag"] == "jumbled_visual_prompt":
                            if conversation[msg_idx]["tag"] == "visual_prompt":
                                vp_mark = self.alternative_visual_prompt_marks[0]
                            else:
                                vp_mark = self.alternative_visual_prompt_marks[1]
                            break
                    
                    mark_container = {
                        "type": "text",
                        "text": vp_mark
                    }
                    if messages and messages[-1]["role"] == "user":
                        if "content" in messages[-1]:
                            messages[-1]["content"].append(mark_container)
                        else:
                            messages[-1]["content"] = [mark_container]
                    else:
                        # if the last message was not from the user, we need to add a new message
                        # to the conversation
                        messages.append(
                            {
                                "role": "user",
                                "content": [mark_container]
                            }
                        )
                
                img_b64 = self._load_img(conversation[msg_idx]["content"])
                img_container = {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                }
                
                if messages and messages[-1]["role"] == "user":
                    if "content" in messages[-1]:
                        messages[-1]["content"].append(img_container)
                    else:
                        messages[-1]["content"] = [img_container]
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": [img_container]
                        }
                    )
                
            elif conversation[msg_idx]["type"] == "text":
                text_container = {
                    "type": "text",
                    "text": conversation[msg_idx]["content"]
                }
                if messages and messages[-1]["role"] == "user":
                    if "content" in messages[-1]:
                        messages[-1]["content"].append(text_container)
                    else:
                        messages[-1]["content"] = [text_container]
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": [text_container]
                        }
                    )
            
            else:
                raise ValueError(f"Unknown message type: {conversation[msg_idx]['type']}")
                
                
        return messages
    
    @retry_with_exponential_backoff
    def generate_content_with_backoff(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)
        
    
    
    def forward(
        self,
        conversation: List[Dict[str, Any]],
        verbose: bool = False,
        **kwargs,
    ):
        
        messages = self.create_prompt(conversation, **kwargs)

        img_url_cnt = 0
        if verbose:
            debug_messages = deepcopy(messages)

            for msg in debug_messages:
                if "content" in msg:
                    if isinstance(msg["content"], list):
                        for part in msg["content"]:
                            if part["type"] in ["image_url"]:
                                part["image_url"]["url"] = f"IMAGE_URL_{img_url_cnt}"
                                img_url_cnt += 1
            print("Input prompt:")
            print(json.dumps(debug_messages, indent=4))

        if self._debug:
            eval_logger.debug("skipping actual API call and returning dummy response for debugging.")
            eval_logger.debug("number of messages in prompt: {}".format(len(messages)))
            eval_logger.debug("number of image URLs in prompt: {}".format(img_url_cnt))
            return "This is a dummy response for debugging purposes."
        
        # import ipdb; ipdb.set_trace()
        response = self.generate_content_with_backoff(
            model=self.model_version,
            messages=messages,
            temperature=self.temperature,
            seed=self.seed,
        )

        return response.choices[0].message.content
    
    
    def post_process_response(self, response):
        return post_process_response(response)
        

def post_process_response(response: str) -> str:
    """
    Post-process the response from the model.
    This function can be customized based on the model's response format.
    """
    # For OpenAI GPT, we can use the Qwen post-process function as a placeholder
    # since it handles similar response structures.
    return qwen_post_process_response(response)
    # In the future, if OpenAI GPT has a different response format, we can implement
    # a specific post-processing function here.    

def main(
    conv_fn: str,
    model_version: Literal["gpt-4.1-2025-04-14", "o4-mini-2025-04-16", "gpt-4o-2024-08-06"] = "gpt-4.1-2025-04-14",
    api_key: str = API_KEY,
    force_media_marks_in_prompt: bool = True,
    video_mark: str = "Here is the video showing the assembly process: ",
    visual_prompt_mark: str = "Here is the visual prompt: ",
    alternative_visual_prompt_marks: List[str] = ["Here is Image A: ", "Here is Image B: "],
    temperature: float = 0.0,
    seed: int = 42,
    debug: bool = False,
    verbose: bool = False,
):
    import yaml
    
    model = OpenAIGPT(
        model_version=model_version,
        api_key=api_key,
        force_media_marks_in_prompt=force_media_marks_in_prompt,
        video_mark=video_mark,
        visual_prompt_mark=visual_prompt_mark,
        alternative_visual_prompt_marks=alternative_visual_prompt_marks,
        temperature=temperature,
        seed=seed,
        debug=debug,
    )

    with open(conv_fn, "r") as f:
        conversation = yaml.safe_load(f)
    
    output = model.forward(conversation, verbose=False)
    print("Response:", output)
    
    post_processed_response = model.post_process_response(output)
    print("Post-processed response:", post_processed_response)
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run OpenAI GPT model on a conversation file.")
    parser.add_argument("--conv_fn", type=str, default=str(root / "src/eval/models/dummy_data/conversation.yaml"), help="Path to the conversation file.")
    parser.add_argument("--model_version", type=str, default="gpt-4.1-2025-04-14", choices=MODEL_VERSIONS, help="Model version to use.")
    parser.add_argument("--force_media_marks_in_prompt", action="store_true", help="Force media marks in prompt.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for response generation.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generation.")
    parser.add_argument("--debug", action="store_true", help="Enable debugging mode. No API calls will be made and dummy responses will be returned.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode for debugging.")

    args = parser.parse_args()
    
    main(
        conv_fn=args.conv_fn,
        model_version=args.model_version,
        force_media_marks_in_prompt=args.force_media_marks_in_prompt,
        temperature=args.temperature,
        seed=args.seed,
        debug=args.debug,
        verbose=args.verbose,
    )