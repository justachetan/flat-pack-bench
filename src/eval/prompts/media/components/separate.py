import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from typing import Dict, Any

import numpy as np
import imageio.v2 as iio
import mediapy as media
from PIL.Image import Image

from src.eval.prompts.media.components.base import MediaComponent

class SeparateMediaComponent(MediaComponent):
    
    def __init__(
        self, 
        **kwargs
    ):
        """SeparateMediaComponent is a media component that combines visual and video components.
        
        This component is designed to handle both visual prompts and video inputs, allowing for flexible media processing.
        """
        super().__init__(**kwargs)

    def get_build_params(self, media: Dict[str, Any], input_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get the parameters used for building the separate media component.
        This can be used to generate a cache key or for logging.
        """
        params = {
            "visual_prompt": media["visual_prompt"],
            "video": media["video"],
            "vp_frame_idxs": media["vp_frame_idxs"],
        }
        
        return params
    
    def get_component_params(self):
        # NOTE: returning empty dict here because for separate
        # components, we do not need to cache the prompt frame and 
        # video again
        return dict()
    
    def get_cache_key(self, media, input_params = None):
        return super().get_cache_key(media, input_params)

    def build(
        self,
        media: Dict[str, Any],
        media_cache_dir: str,
        override_cache: bool = False,
        **kwargs
    ) -> dict:
        """
        Process the visual prompt and video input and return a dictionary containing both components.
        Args:
            media (Dict[str, Any]): A dictionary containing the visual prompt and video components.
                Expected keys are 'visual_prompt' and 'video'.
            media_cache_dir (str): Directory where the processed media will be cached.
        Returns:
            dict: A dictionary containing the processed visual prompt and video frames.
        """
        # NOTE: no need for caching or checking the cache here, 
        # as no new media is generated
        return media
        
        
