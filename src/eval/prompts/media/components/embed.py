import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from typing import Dict, Any

import os
import os.path as osp
import copy
import numpy as np
import imageio.v2 as iio
import mediapy
from PIL.Image import Image

from src.eval.prompts.media.components.base import MediaComponent


class EmbedComponent(MediaComponent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_build_params(self, media: Dict[str, Any], input_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get the parameters used for building the embed component.
        This can be used to generate a cache key or for logging.
        """
        params = {
            "video": media["video"],
            "visual_prompts": media["visual_prompts"],
            "vp_frame_idxs": media["vp_frame_idxs"],
        }
        
        return params
    
    def get_component_params(self):
        # NOTE: returning a dummy dict here for cache key generation
        
        return {
            "component_type": "embed"
        }
    
    def get_cache_key(self, media: Dict[str, Any], input_params: Dict[str, Any] = None) -> str:
        """
        Generate a cache file name based on parameters and build arguments.
        This is used to avoid recomputing the same media component.
        """
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
            media_cache_dir (str): Directory where the processed media will be cached.
            override_cache (bool, optional): Whether to override cached media. Defaults to False.
        Returns:
            dict: A dictionary containing the processed media.
        """
        
        cache_fn = self.get_cache_key(media)
        if osp.exists(osp.join(media_cache_dir, f"{cache_fn}.mp4")) and (not override_cache):
            embedded_media = copy.deepcopy(media)
            embedded_media["video"] = f"{cache_fn}.mp4"
            return embedded_media
            
        embedded_media = copy.deepcopy(media)
        video = mediapy.read_video(osp.join(media_cache_dir, media["video"]))
        for frame_idx in media["vp_frame_idxs"]:
            video[frame_idx] = iio.imread(
                osp.join(media_cache_dir, media["visual_prompts"][frame_idx])
            )
        
        mediapy.write_video(
            osp.join(media_cache_dir, f"{cache_fn}.mp4"), 
            video, fps=1
        )
        embedded_media["video"] = f"{cache_fn}.mp4"
        return embedded_media
    
def main(video_path: str, question_yaml: str, img_dir: str, mask_dir: str):
    from src.eval.prompts.media.components.video import VideoComponent
    from src.eval.prompts.templates.questions.convert_yaml_to_json import convert_yaml_to_json
    from src.eval.prompts.media.components.visual_prompt import VisualPrompt
    
    media_cache_dir = osp.join(root, "tmp", "media_cache")
    os.makedirs(media_cache_dir, exist_ok=True)
    video_component = VideoComponent(media_cache_dir=media_cache_dir, resolution=(480, 640))
    media = video_component.build(video_path, {})
    print("after video:")
    print(media)
    
    question_json = convert_yaml_to_json(question_yaml)
    category = question_json["vid_category"]
    name = question_json["furniture_name"]
    vid = question_json["video_id"]
    frame_idxs = question_json["frame_idx"]
    jumble_map = question_json.get("jumble_map", None)
    
    resolution = (480, 640)  # H x W
    img_dir = osp.join(img_dir, category, name, vid)
    mask_path = osp.join(mask_dir, category, name, vid, f"{vid}.json")
    colors = None
    edge_colors = None

    visual_prompt_component = VisualPrompt(
        media_cache_dir=media_cache_dir,
        resolution=resolution,
    )
    media = visual_prompt_component.build(
        img_dir=img_dir,
        mask_path=mask_path,
        frame_idxs=frame_idxs,
        colors = colors,
        edge_colors=edge_colors,
        jumble_map=jumble_map,
        media=media,
    )
    print("after visual prompts:")
    print(media)
    
    
    embed_component = EmbedComponent(media_cache_dir=media_cache_dir)
    embedded_media = embed_component.build(media)
    print("after embedding:")
    print(embedded_media)
    
if __name__ == "__main__":
    """
    python3 embed.py \
        --video_path data/videos/keyframe-video/fps-1/Chair/vedbo/NdkuJ9cwOuE/NdkuJ9cwOuE.mp4 \
        --question_yaml data/questions/yamls/098.yaml \
        --img_dir data/rgb-frames/ \
        --mask_dir data/segmentation-masks
    """
    import argparse
    parser = argparse.ArgumentParser(description="Embed media component")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file")
    parser.add_argument("--question_yaml", type=str, required=True, help="Path to the question YAML file")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory containing masks")
    
    args = parser.parse_args()
    main(args.video_path, args.question_yaml, args.img_dir, args.mask_dir)