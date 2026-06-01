import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from typing import Dict, Any, Literal, List

import os
import os.path as osp
import cv2
import copy
import numpy as np
import imageio.v2 as iio
import mediapy 
from PIL import Image, ImageDraw, ImageFont

import torch
from torchvision.utils import make_grid

from src.eval.prompts.media.components.base import MediaComponent


class NumberIt(MediaComponent):
    
    def __init__(
        self,
        ignore_subspl: bool= True,
        font_size: int = 40,
        font_color: str = "red",
        font_path: str = None,
        zero_index: bool = True,
        **kwargs
    ):
        """Number the frames in the video. Similar to https://arxiv.org/abs/2411.10332

        Args:
            ignore_subspl (bool, optional): ignore sub-sampling that may have occured in the video. Defaults to True.
        """
        super().__init__(**kwargs)
        self.ignore_subspl = ignore_subspl
        self.font_size = font_size
        self.font_color = font_color
        self.font_path = font_path
        if font_path:
            try:
                self.font = ImageFont.truetype(font_path, font_size)
            except OSError:
                self.font = ImageFont.load_default()
        else:
            self.font = ImageFont.load_default()
        self.zero_index = zero_index

    def _number_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
    ):
        """number the frame with the frame index.

        Args:
            frame (Image): np.ndarray of the frame.
            frame_idx (int): Index of the frame to be numbered.

        Returns:
            Image: PIL Image of the numbered frame.
        """        
        
        # Add frame number
        numbered_frame = Image.fromarray(frame)
        draw = ImageDraw.Draw(numbered_frame)
        
        # Calculate text position
        width, height = numbered_frame.size
        text = str(frame_idx)
        text_bbox = draw.textbbox((0, 0), text, font=self.font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        x = width - text_width
        y = height - text_height - text_height/3
        
        # TODO: font color is not coming out correctly. FIX
        draw.text((x, y), text, font=self.font, fill=self.font_color)
        # final_frame = cv2.cvtColor(np.array(numbered_frame), cv2.COLOR_RGB2BGR)
        # final_frame = Image.fromarray(final_frame)   
        return numbered_frame
        
    def get_build_params(self, media: Dict[str, Any], input_params: Dict[str, Any]=None) -> List[str]:
        frame_idxs = media.get("frame_idxs", None)
        build_params = {
            "video": media["video"],
        }
        if frame_idxs is not None:
            build_params["frame_idxs"] = frame_idxs
        return build_params
    
    def get_component_params(self):
        return {
            "ignore_subspl": self.ignore_subspl,
            "font_size": self.font_size,
            "font_color": self.font_color,
            "font_path": self.font_path,
            "zero_index": self.zero_index,
        }
        
    def get_cache_key(self, media: Dict[str, Any], input_params: Dict[str, Any] = None) -> str:
        return super().get_cache_key(media, input_params)
    
    def build(
        self,
        media: Dict[str, Any],
        media_cache_dir: str,
        override_cache: bool = False,
        **kwargs,
    ):
        """
        Process the media input and return a dictionary containing the processed media.
        
        Args:
            media (Dict[str, Any]): A dictionary containing the media components.
            frame_idxs (List[int], optional): List of frame indices in the video.
                Required if `ignore_subspl` is False, otherwise ignored.
            media_cache_dir (str): Directory to cache the processed media.
            override_cache (bool, optional): Whether to override the cached media. Defaults to False.
        Returns:
            dict: A dictionary containing the processed media.
        """
        if "video" not in media or not isinstance(media["video"], str):
            raise ValueError("Media must contain a 'video' key with a string filename.")
        
        video_fn = osp.join(media_cache_dir, media["video"])
        
        if self.ignore_subspl or "frame_idxs" not in media:
            video = mediapy.read_video(video_fn)
            frame_idxs = list(range(video.shape[0]))
        else:
            frame_idxs = media["frame_idxs"]
            assert len(frame_idxs) == video.shape[0], \
                "Frame indices length must match the number of frames in the video."
        
        
        
        cache_fn = self.get_cache_key(media)
        if osp.exists(osp.join(media_cache_dir, f"{cache_fn}.mp4")) and (not override_cache):
            return {
                "video": cache_fn,
                "frame_idxs": frame_idxs,
            }
        
        video = mediapy.read_video(osp.join(media_cache_dir, video_fn))
        numbered_video = copy.deepcopy(video) 
        for frame_idx in frame_idxs:
            numbered_video[frame_idx] = self._number_frame(
                frame=video[frame_idx],
                frame_idx=frame_idx if self.zero_index else frame_idx + 1,
            )
            
        
        mediapy.write_video(
            osp.join(media_cache_dir, f"{cache_fn}.mp4"), 
            numbered_video, fps=1
        )

        numbered_media = {
            "video": f"{cache_fn}.mp4",
            "frame_idxs": frame_idxs
        }
        
        return numbered_media
        
        
def main(video_path: str):
    from src.eval.prompts.media.components.video import VideoComponent
    media_cache_dir = osp.join(root, "tmp", "media_cache")
    os.makedirs(media_cache_dir, exist_ok=True)
    video_component = VideoComponent(media_cache_dir=media_cache_dir, resolution=(480, 640))
    media = video_component.build(video_path, {})
    print("before numbering:")
    print(media)
    
    numbered_media = NumberIt(
        media_cache_dir=media_cache_dir,
        ignore_subspl=True,
        font_size=40,
        font_color="red",
        font_path=None,
        zero_index=False,
    )
    numbered_media = numbered_media.build(media)
    print("after numbering:")
    print(numbered_media)
    
if __name__ == "__main__":
    # python3 video.py path/to/video.mp4
    # ex. python3 video.py data/videos/keyframe-video/fps-1/Bench/applaro/KPs0ik2FcsY/KPs0ik2FcsY.mp4
    import argparse
    parser = argparse.ArgumentParser(description="Number frames in a video.")
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    args = parser.parse_args()
    
    main(args.video_path)
