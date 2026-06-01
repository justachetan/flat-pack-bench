import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from typing import Tuple, Literal, Dict, Any
import os
import copy
import os.path as osp
import imageio.v2 as iio
import mediapy

from src.eval.prompts.media.components.base import MediaComponent

class VideoComponent(MediaComponent):
    def __init__(self, 
            resolution: Tuple[int, int] = (480, 640),
            **kwargs):
        """VideoComponent is a media component for processing video files.

        Args:
            resolution (Tuple[int, int], optional): The resolution of the video. Defaults to (640, 480).
            tmp_dir (str, optional): Temporary directory for processing. Defaults to None.
        """
        super().__init__(**kwargs)
        self.resolution = resolution
        
    def get_build_params(self, media: Dict[str, Any], input_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get the parameters used for building the video component.
        This can be used to generate a cache key or for logging.
        """
        params = {
            "video_path": input_params["video_path"],
        }
        
        return params
    
    def get_component_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the video component.
        This can be used to generate a cache key or for logging.
        """
        return {
            "resolution": self.resolution,
        }

    def get_cache_key(self, media: Dict[str, Any], input_params: Dict[str, Any] = None) -> str:
        return super().get_cache_key(media, input_params)

    def build(self, video_path: str, media: Dict[str, Any], media_cache_dir: str, override_cache: bool = False, **kwargs) -> dict:
        """
        Process the video input and return the video
        component depending on the specified format.
        
        Args:
            video_path (str): Path to the video file.
        Returns:
            Dict[str, np.ndarray]: A dictionary containing the processed video frames.
        """
        # T x H x W x C
        cache_fn = self.get_cache_key(media, {"video_path": video_path})
        if osp.exists(osp.join(media_cache_dir, f"{cache_fn}.mp4")) and (not override_cache):
            new_media = copy.deepcopy(media)
            new_media["video"] = f"{cache_fn}.mp4"
            return new_media
            
        frames = mediapy.read_video(video_path,)
        if self.resolution is not None and self.resolution != frames.shape[1:3]:
            frames = mediapy.resize_video(frames, self.resolution)

        mediapy.write_video(
            osp.join(media_cache_dir, f"{cache_fn}.mp4"),
            frames, fps=1
        )
        if media is None:
            media = dict()
        media.update({"video": f"{cache_fn}.mp4"})
        return media
        
def main(video_path: str):
    media_cache_dir = osp.join(root, "tmp", "media_cache")
    os.makedirs(media_cache_dir, exist_ok=True)
    video_component = VideoComponent(media_cache_dir=media_cache_dir, resolution=(480, 640))
    media = video_component.build(video_path, {})
    print(media)
    
if __name__ == "__main__":
    # python3 video.py path/to/video.mp4
    # ex. python3 video.py data/videos/keyframe-video/fps-1/Bench/applaro/KPs0ik2FcsY/KPs0ik2FcsY.mp4
    import argparse
    parser = argparse.ArgumentParser(description="Process a video file.")
    parser.add_argument("video_path", type=str, help="Path to the video file.")
    args = parser.parse_args()
    main(args.video_path)
        