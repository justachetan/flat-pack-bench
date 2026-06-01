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


class SubSampler(MediaComponent):
    
    def __init__(
        self,
        max_num_frames: int = 32,
        retain_vp_frames: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_num_frames = max_num_frames
        self.retain_vp_frames = retain_vp_frames

    def _subsample(self, 
                   video: np.ndarray,
                   retain_frame_idxs: List[int]=None
        ) -> np.ndarray:
        """Subsample frames from a video to a maximum number of frames.
        If retain_frame_idxs is provided, retain those frame indices in the place of
        the closest evenly spaced frames.

        Args:
            video (np.ndarray): video frames as a numpy array of shape (T, H, W, C).
            retain_frame_idxs (List[int], optional): frame indices to retain. Defaults to None.

        Returns:
            np.ndarray: subsampled video frames.
        """
        frame_idxs_to_use = np.linspace(
            0, video.shape[0]-1,
            num=self.max_num_frames, dtype=int,
        )
        
        final_frame_idxs_to_use = None
        # Combine evenly spaced indices with those to retain
        if retain_frame_idxs:
            retain_arr = np.array(retain_frame_idxs, dtype=int)
            combined = np.unique(np.concatenate([frame_idxs_to_use, retain_arr]))
            final_frame_idxs_to_use = np.sort(combined)[: self.max_num_frames]
        else:
            final_frame_idxs_to_use = frame_idxs_to_use
        subsampled_video = video[final_frame_idxs_to_use]
        return subsampled_video, final_frame_idxs_to_use

    def get_build_params(self, media: Dict[str, Any], input_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get the parameters used for building the subsampler component.
        This can be used to generate a cache key or for logging.
        """
        return media
    
    def get_component_params(self):
        return {
            "max_num_frames": self.max_num_frames,
            "retain_vp_frames": self.retain_vp_frames,
        }
    
    def get_cache_key(self, media, input_params = None):
        return super().get_cache_key(media, input_params)
    
    def build(
        self,
        media: Dict[str, Any],
        media_cache_dir: str,
        override_cache: bool = False,
        **kwargs,
    ):
        if "video" not in media or not isinstance(media["video"], np.ndarray):
            raise ValueError("Media must contain a 'video' key with a numpy array.")

        retain_frame_idxs = media.get(
            "vp_frame_idxs", list(media.get("visual_prompts").keys())
        )
        
        cache_fn = self.get_cache_key(media)
        if osp.exists(osp.join(media_cache_dir, f"{cache_fn}.mp4")) and (not override_cache):
            subsampled_media = {key: media[key] for key in media if key != "frame_idxs"}
            subsampled_media["video"] = cache_fn
            return subsampled_media
            
        video = mediapy.read_video(media["video"])
        subsampled_video, frame_idxs = self._subsample(video, retain_frame_idxs)
        
        mediapy.write_video(
            osp.join(media_cache_dir, cache_fn), 
            subsampled_video, fps=1
        )
        subsampled_media = copy.deepcopy(media)
        subsampled_media["video"] = cache_fn
        subsampled_media["frame_idxs"] = frame_idxs
        return subsampled_media