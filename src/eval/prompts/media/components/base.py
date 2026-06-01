import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
# src/components/base.py
from typing import Dict, Any
import json
from abc import ABC, abstractmethod
from omegaconf import OmegaConf

from src.eval.prompts.templates.questions.common import calculate_sha256

class MediaComponent(ABC):
    def __init__(self, media_cache_dir: str="/tmp/media_cache", **kwargs):
        self.params = kwargs
        self.media_cache_dir = media_cache_dir

    @abstractmethod
    def build(self, media: Dict[str, Any], media_cache_dir: str, **kwargs) -> dict:
        """
        Process inputs (video/image), return a dict mapping
        placeholder names → file paths or Markdown embeds.
        e.g. { "image": "…/frame_37.png" }
        """
        pass

    @abstractmethod
    def get_build_params(self, media: Dict[str, Any], input_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get the parameters used for building the media component.
        This can be used to generate a cache key or for logging.
        """
        pass
    
    @abstractmethod
    def get_component_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the media component.
        This can be used to generate a cache key or for logging.
        """
        pass
    
    def get_cache_key(self, media: Dict[str, Any], input_params: Dict[str, Any] = None) -> str:
        """
        Generate a cache file name based on parameters and build arguments.
        This is used to avoid recomputing the same media component.
        """
        build_args = self.get_build_params(media, input_params)
        build_args.update(self.get_component_params())
        for k, v in build_args.items():
            if OmegaConf.is_config(v):
                build_args[k] = OmegaConf.to_container(v, resolve=True)
        # import ipdb; ipdb.set_trace()
        cache_key = json.dumps(build_args, sort_keys=True)
        cache_key = calculate_sha256(cache_key)
        return cache_key