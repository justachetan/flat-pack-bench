import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from typing import Literal, Dict, Any, List

import os
import os.path as osp
import time
import glob
import json
from dotenv import load_dotenv
load_dotenv()

import mediapy
import numpy as np
from PIL import Image

import textwrap

from gens import gens_frame_sampler, setup_model

from src.eval.models.base_model import BaseModel
from src.eval.models.gemini import Gemini

class GenFS(BaseModel):
    def __init__(
        self, 
        model_name: Literal["yaolily/GenS", "yaolily/GenS-qwen2d5-vl-3b"] = "yaolily/GenS", 
        attn_implementation: Literal["flash_attention_2", "eager"] = "flash_attention_2",
        relevance_threshold: int = 4,
        base_model_name: Literal["gemini-2.5-flash", "gemini-2.5-pro"] = "gemini-2.5-pro",
        base_api_key: str = None,
        base_thinking_config: Dict[str, Any] = {"thinking_budget": -1, "include_thoughts": True},
        base_generate_config: Dict[str, Any] = {"temperature": 0.0, "media_resolution": "high", "seed": 42},
        base_use_vertexai: bool = False,
        base_gs_bucket: str = "vlm_4d_bench_vids",
        base_video_fps: int = None,
        base_api_version: str = "v1",
        base_response_schema: str | None = "GeminiAnswer",
        base_tmpdir: str = "tmp/gemini",
        base_clear_cache_at_init: bool = True,
        use_paid_api: bool = False,
        handle_concat_videos_specially: bool = False,
        **kwargs
    ):
        """
        NOTE: default behavior is greedy (do_sample == False), hence no temperature parameter here.
        
        For now, I am hard-setting the base model to Gemini-2.5-Pro.
        Later, we can add more options.
        """
        super().__init__(**kwargs)
        self.model_name = model_name
        self.attn_implementation = attn_implementation
        
        self.model, self.tokenizer, self.processor = setup_model(self.model_name, 
                                                                 attn_implementation=self.attn_implementation)

        self.relevance_threshold = relevance_threshold
        if self.relevance_threshold < 1 or self.relevance_threshold > 5:
            raise ValueError("relevance_threshold should be between 1 and 5.")

        
        os.makedirs(base_tmpdir, exist_ok=True)    
        self.base_model = Gemini(
            model_name=base_model_name,
            api_key=base_api_key,
            thinking_config=base_thinking_config,
            generate_config=base_generate_config,
            use_vertexai=base_use_vertexai,
            gs_bucket=base_gs_bucket,
            video_fps=base_video_fps,
            api_version=base_api_version,
            response_schema=base_response_schema,
            tmpdir=base_tmpdir,
            clear_cache_at_init=base_clear_cache_at_init,
            use_paid_api=use_paid_api,
        )
        self.handle_concat_videos_specially = handle_concat_videos_specially
        self._chunksize = 256
        
    def create_prompt(
        self,
        conversation: List[Dict[str, Any]],
    ):
        """
        NOTE: for genFS, we only consider video-only 
        prompts for now.
        
        """
        text = ""
        video_fn = None
        for msg_idx, msg in enumerate(conversation):
            msg_type = msg["type"]
            video_tag = "video"
            if msg_type == "video":
                video_fn = msg["content"]
                video_tag = msg.get("tag", "video")
            elif msg_type == "text":
                text += msg["content"] + "\n"
            else:
                continue
                # raise ValueError(f"Unsupported message type: {msg_type}")
            
        return text.strip(), video_fn, video_tag
    
    def _handle_video_input(self, video_path: str):
        if osp.isdir(video_path):
            frame_fns = sorted(glob.glob(osp.join(video_path, "*")), key=lambda x: osp.basename(x).split('.')[0])
            frames = frame_fns
        else:
            frames = mediapy.read_video(video_path)
            frames = [Image.fromarray(frame) for frame in frames]
            
        return frames    

    def _process_relevant_frames(
            self, 
            frames: List[Image.Image], 
            relevance_scores: Dict[str, int], 
            out_video_fn: str = None,
            offset: int = 0
        ):
        """
        Process frames based on relevance scores.
        
        Args:
            frames (List[Image.Image]): list of video frames.
            relevance_scores (Dict[str, int]): dict mapping frame indices to relevance scores.
            offset (int): offset frames consisting of visual prompts added to the clip's beginning.
                These frames should be ignored when applying relevance scores, since they are not part of the original video.
        
        Returns:
            processed_frames (List[Image.Image]): list of processed frames.
        """
        # Example processing: filter frames based on a relevance threshold
        indices_to_keep = list()
        for frame_idx_str, score in relevance_scores.items():
            if isinstance(frame_idx_str, str) and "-" in frame_idx_str:
                start_idx, end_idx = map(int, frame_idx_str.split("-"))
                if start_idx < offset:
                    start_idx = offset
                if end_idx < offset:
                    continue
                if score >= self.relevance_threshold:
                    indices_to_keep.extend(list(range(start_idx, end_idx + 1)))
            else:
                frame_idx = int(frame_idx_str)
                if frame_idx < offset:
                    continue
                if score >= self.relevance_threshold:
                    indices_to_keep.append(frame_idx)
        indices_to_keep = sorted(indices_to_keep)
        processed_frames = [np.array(frame) for idx, frame in enumerate(frames) if idx in indices_to_keep]
        
        
        return processed_frames
    
    def forward(
        self,
        conversation: List[Dict[str, Any]],
    ):
        """
        Args:
            conversation (List[Dict[str, Any]]): list of messages in the conversation.
                Each message is a dict with keys "type" and "content".
                "type" can be "text" or "video".
                "content" is the actual content of the message.
        
        Returns:
            response (str): generated response text.
        """
        prompt_text, video_fn, video_tag = self.create_prompt(conversation)
        
        frames = self._handle_video_input(video_fn)

        filtered_frames = []
        all_relevance_scores = dict()

        offset = 0
        if video_tag.startswith("concat") and self.handle_concat_videos_specially:
            offset = 1            
            if "tracking" in video_tag:
                offset = 2
        new_chunksize = self._chunksize - offset
        frame_iter = enumerate(range(offset, len(frames), new_chunksize))

        filtered_frames = frames[:offset]  # start with offset frames (visual prompts)
        for clip_idx, start_frame_idx in frame_iter:
            # import ipdb; ipdb.set_trace()
            # Process video
            frame_chunk_idx = np.r_[0:offset, start_frame_idx:start_frame_idx+new_chunksize]
            frame_chunk = [frames[i] for i in frame_chunk_idx if i < len(frames)]
            response = gens_frame_sampler(prompt_text, frame_chunk, 
                                            self.model, self.tokenizer, self.processor,
                                            max_new_tokens=1024)
            # some weird things: not all frames are labeled, and not all ratings are present
            
            # do further processing and forward to base LLM
            try:
                relevance_scores = json.loads(response)
            except json.JSONDecodeError:
                relevance_scores = {}
            if len(relevance_scores) == 0:
                # if no relevance scores are found, assume all frames are highly relevant
                relevance_scores = {str(idx): 5 for idx in range(self._chunksize)}
            processed_frames = self._process_relevant_frames(
                frame_chunk, 
                relevance_scores, 
                offset=offset
            )
            filtered_frames.extend(processed_frames)
            all_relevance_scores[clip_idx] = relevance_scores # relevance scores first two indices will always be for visual prompts

        if len(filtered_frames) == 0:
            # if no frames are selected, keep the original frames
            filtered_frames.extend([np.array(frame) for frame in frames])

        processed_video_fn = osp.join(self.base_model.tmpdir, f"{osp.basename(video_fn).split('.')[0]}_filtered.mp4")
        mediapy.write_video(processed_video_fn, filtered_frames, fps=1)

        # prepare new conversation for base model
        new_conversation = []
        for msg in conversation:
            if msg["type"] == "video":
                new_conversation.append({
                    "type": "video",
                    "tag": video_tag,
                    "content": processed_video_fn
                })
            else:
                new_conversation.append(msg)
        base_model_response = self.base_model.forward(new_conversation)
        base_model_response["relevance_scores"] = all_relevance_scores
        return base_model_response


    def post_process_response(self, response):
        return self.base_model.post_process_response(response)
    
if __name__ == "__main__":

    """
    Collage video testing:
    """

    import argparse
    import yaml
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--conv-fn", type=str, default=str(root / "src/eval/models/dummy_data/conversation.yaml"))

    args = parser.parse_args()
    conv_fn = args.conv_fn
    with open(conv_fn, 'r') as f:
        conversation = yaml.safe_load(f)
    # simple test
    model = GenFS(model_name="yaolily/GenS", base_clear_cache_at_init=False, use_paid_api=True)

    response = model.forward(conversation)
    print("Response:", response)
