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

from decord import VideoReader
import torch
import numpy as np
from PIL import Image
from transformers import (
    AriaProcessor, AriaForConditionalGeneration
)
from tqdm import tqdm

from src.eval.models.base_model import BaseModel

class Aria(BaseModel):
    def __init__(
        self,
        model_name: Literal[
            "rhymes-ai/Aria",
        ] = "rhymes-ai/Aria",
        change_sys_prompt: bool = False,
        temperature: float = 0.0,
        do_sample: bool = False,
        dtype: str = "bfloat16",
        max_new_tokens: int = 2048,
        num_frames: int=100,
        cache_dir: str="tmp/aria",
        max_image_size: int=640,
        **kwargs: Any,
    ):
        """Aria model wrapper.

        Args:
            model_name (Literal[ &quot;llava, optional): model version name. Defaults to "rhymes-ai/Aria".
            change_sys_prompt (bool, optional): edit the system prompt. Defaults to False.
            temperature (float, optional): temperature for generation. Defaults to 0.0.
            do_sample (bool, optional): enable random generation. False means greedy decoding. Defaults to False.
            dtype (str, optional): data type for the model weights. Defaults to "bfloat16".
            max_new_tokens (int, optional): maximum number of tokens to generate. Defaults to 2048.
            cache_dir (str): directory the cached video frames are stored in.
            **kwargs: Any additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.change_sys_prompt = change_sys_prompt
        self.temperature = temperature
        self.do_sample = do_sample
        self.max_new_tokens = max_new_tokens
        self.num_frames = num_frames
        self.cache_dir = cache_dir
        self.max_image_size = max_image_size

        self.model = AriaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        self.processor = AriaProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

    def resize_image(
            self,
            img,
            max_image_size=None
    ):
        if max_image_size is None:
            max_image_size = self.max_image_size
        
        width, height = img.size
        print(img.size)
        if width > height:
            scale_factor = max_image_size / width
        else:
            scale_factor = max_image_size / height
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        return img.resize((new_width, new_height))



    def load_video(
            self,
            video_file,
            num_frames=None,
            cache_dir=None,
            verbosity="DEBUG"
    ):
        if num_frames is None:
            num_frames = self.num_frames
        if cache_dir is None:
            cache_dir = self.cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)

        video_basename = os.path.basename(video_file)
        cache_subdir = os.path.join(cache_dir, f"{video_basename}_{num_frames}")
        os.makedirs(cache_subdir, exist_ok=True)

        cached_frames = []
        missing_frames = []
        frame_indices = []
        
        for i in range(num_frames):
            frame_path = os.path.join(cache_subdir, f"frame_{i}.jpg")
            if os.path.exists(frame_path):
                cached_frames.append(frame_path)
            else:
                missing_frames.append(i)
                frame_indices.append(i) 
                
        vr = VideoReader(video_file)
        duration = len(vr)
        fps = vr.get_avg_fps()
                
        frame_timestamps = [int(duration / num_frames * (i+0.5)) / fps for i in range(num_frames)]
        
        if verbosity == "DEBUG":
            print("Already cached {}/{} frames for video {}, enjoy speed!".format(len(cached_frames), num_frames, video_file))
        # If all frames are cached, load them directly
        if not missing_frames:
            return [Image.open(frame_path).convert("RGB") for frame_path in cached_frames], frame_timestamps

        

        actual_frame_indices = [int(duration / num_frames * (i+0.5)) for i in missing_frames]


        missing_frames_data = vr.get_batch(actual_frame_indices).asnumpy()

        for idx, frame_index in enumerate(tqdm(missing_frames, desc="Caching rest frames")):
            img = Image.fromarray(missing_frames_data[idx]).convert("RGB")
            img = self.resize_image(img)
            frame_path = os.path.join(cache_subdir, f"frame_{frame_index}.jpg")
            img.save(frame_path)
            cached_frames.append(frame_path)

        cached_frames.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
        return [Image.open(frame_path).convert("RGB") for frame_path in cached_frames], frame_timestamps

    def create_image_gallery(
            self,
            images,
            columns=3,
            spacing=20,
            bg_color=(200, 200, 200)
    ):
        """
        Combine multiple images into a single larger image in a grid format.
        
        Parameters:
            image_paths (list of str): List of file paths to the images to display.
            columns (int): Number of columns in the gallery.
            spacing (int): Space (in pixels) between the images in the gallery.
            bg_color (tuple): Background color of the gallery (R, G, B).
        
        Returns:
            PIL.Image: A single combined image.
        """
        # Open all images and get their sizes
        img_width, img_height = images[0].size  # Assuming all images are of the same size

        # Calculate rows needed for the gallery
        rows = (len(images) + columns - 1) // columns

        # Calculate the size of the final gallery image
        gallery_width = columns * img_width + (columns - 1) * spacing
        gallery_height = rows * img_height + (rows - 1) * spacing

        # Create a new image with the calculated size and background color
        gallery_image = Image.new('RGB', (gallery_width, gallery_height), bg_color)

        # Paste each image into the gallery
        for index, img in enumerate(images):
            row = index // columns
            col = index % columns

            x = col * (img_width + spacing)
            y = row * (img_height + spacing)

            gallery_image.paste(img, (x, y))

        return gallery_image


    def get_placeholders_for_videos(
            self,
            frames: List,
            timestamps=[]
    ):
        contents = []
        if not timestamps:
            for i, _ in enumerate(frames):
                contents.append({"text": None, "type": "image"})
            contents.append({"text": "\n", "type": "text"})
        else:
            for i, (_, ts) in enumerate(zip(frames, timestamps)):
                contents.extend(
                    [
                        {"text": f"[{int(ts)//60:02d}:{int(ts)%60:02d}]", "type": "text"},
                        {"text": None, "type": "image"},
                        {"text": "\n", "type": "text"}
                    ]
                )
        return contents

    def create_prompt(
        self,
        conversation: List[Dict[str, Any]],
        **kwargs,
    ):
        
        messages = []
        images = []
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
                video_path = conversation[msg_idx]["content"]
                frames, frame_timestamps = self.load_video(video_path)
                contents = self.get_placeholders_for_videos(frames, frame_timestamps)
                input_msg = {
                    "role": "user",
                    "content": contents
                }
                images += frames
                
            elif conversation[msg_idx]["type"] == "text":
                input_msg = {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": conversation[msg_idx]["content"]
                    }]
                }
                
            elif conversation[msg_idx]["type"] == "image":
                image_path = conversation[msg_idx]["content"]
                input_msg = {
                    "role": "user",
                    "content": [{
                        "type": "image",
                        "image": image_path,
                    }]
                }
                images += [self.resize_image(Image.open(image_path).convert("RGB"))]
            else:
                raise ValueError(f"Unknown message type: {conversation[msg_idx]['type']}")
                
            if len(messages) == 0 or messages[-1]["role"] != "user":
                messages.append(input_msg)
            else:
                messages[-1]["content"].extend(input_msg["content"])
                
        return messages, images
    
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
        messages, images = self.create_prompt(conversation, **kwargs)
        if verbose:
            print(f"Messages:\n{json.dumps(messages, indent=2)}\n")
            input_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )
            print(f"Input prompt:\n{input_prompt}\n")

        
        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )

        inputs = self.processor(text=text, images=images, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(self.model.dtype)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "stop_strings": ["<|im_end|>"],
            "tokenizer": self.processor.tokenizer,
            "do_sample": self.do_sample,
            "temperature": self.temperature
        }
        if not self.do_sample:
            generation_kwargs.pop("temperature", None)

        output_ids = self.model.generate(
            **inputs,
            **generation_kwargs
        )

        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = self.model.generate(
                **inputs,
                **generation_kwargs
            )
            output_ids = output[0][inputs["input_ids"].shape[1]:]
            result = self.processor.decode(output_ids, skip_special_tokens=True)
        
        return result
    
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
    model_name: str = "rhymes-ai/Aria",
    change_sys_prompt: bool = False,
    **kwargs: Any
):
    import time
    import yaml
    
    model = Aria(
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
    parser = argparse.ArgumentParser(description="Aria Model Inference")
    parser.add_argument("--conv_fn", type=str, default=str(root / "src/eval/models/dummy_data/conversation.yaml"), help="Path to the conversation YAML file.")
    parser.add_argument("--model_name", type=str, default="rhymes-ai/Aria", 
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
