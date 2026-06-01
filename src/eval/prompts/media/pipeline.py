import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from typing import Tuple, Literal, List, Any, Dict
import os
import os.path as osp

from dataclasses import dataclass

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra import initialize, compose

from src.eval.prompts.media.components import (
    visual_prompt,
    video,
    separate,
    embed,
    subspl,
    collage,
    concat,
)
from src.eval.prompts.media.components.base import MediaComponent
from src.eval.prompts.templates.questions.common import calculate_sha256

@dataclass
class MediaComponentRegistry:
    
    """Registry for media components."""
    video: video.VideoComponent
    visual_prompt: visual_prompt.VisualPrompt
    separate: separate.SeparateMediaComponent
    embed: embed.EmbedComponent
    subspl: subspl.SubSampler
    collage: collage.CollageComponent
    concat: concat.ConcatComponent

class MediaPipeline:
    
    def __init__(self, cfg: DictConfig, media_cache_dir: str):
        """
        Args:
            cfg: Hydra-loaded config, must contain `media_pipeline.pipeline`: a list of step configs.
            media_cache_dir: Directory to cache media files.
        """
        self.cfg = cfg
        # Instantiate each component from its config
        self.media_cache_dir = media_cache_dir
        self.steps = [
            instantiate(step_cfg) 
            for step_cfg in cfg.pipeline
        ]
        
    def run(
        self,
        video_path: str,
        img_dir: str,
        mask_path: str,
        frame_idxs: List[int],
        jumble_map: Dict[str, str] = None,
        colors: Any = None,
        edge_colors: Any = None,
        override_cache: bool = False,
        **kwargs
    ):
        """Run the media pipeline.

        Args:
            video_path (str): Path to the input video file.
            img_dir (str): Directory containing images for visual prompts.
            mask_path (str): Path to the JSON file containing masks.
            frame_idxs (List[int]): List of frame indices to use.
            jumble_map (Dict[str, str], optional): Mapping for jumbling parts. Defaults to None.
            colors (Any, optional): Colors for visualization. Defaults to None.
            edge_colors (Any, optional): Edge colors for visualization. Defaults to None.
            override_cache (bool, optional): Whether to override cached media. Defaults to False.
            **kwargs: Additional keyword arguments for specific components.
        """

        os.makedirs(self.media_cache_dir, exist_ok=True)
        media = dict()
        for step_idx, step in enumerate(self.steps):
            
            media = step.build(
                video_path=video_path,
                img_dir=img_dir,
                mask_path=mask_path,
                frame_idxs=frame_idxs,
                colors=colors,
                edge_colors=edge_colors,
                media=media,
                jumble_map=jumble_map,
                media_cache_dir=self.media_cache_dir,
                override_cache=override_cache,
                **kwargs
            )
            
        return media

def main(
    video_dir: str, question_yaml: str, img_dir: str, mask_dir: str,
    pipeline: str, media_cache_dir: str
):
    from src.eval.prompts.templates.questions.convert_yaml_to_json import convert_yaml_to_json

    question_json = convert_yaml_to_json(question_yaml)
    category = question_json["vid_category"]
    name = question_json["furniture_name"]
    vid = question_json["video_id"]
    frame_idxs = question_json["frame_idx"]
    if isinstance(frame_idxs, int):
        frame_idxs = [frame_idxs]
    jumble_map = question_json.get("jumble_map", None)
    
    video_path = osp.join(video_dir, category, name, vid, f"{vid}.mp4")
    mask_path = osp.join(mask_dir, category, name, vid, f"{vid}.json")
    img_dir = osp.join(img_dir, category, name, vid)

    # 1) Initialize Hydra (point it at your conf directory)
    initialize(config_path="../../configs", version_base="1.1", job_name="manual_test")
    # 2) Compose the config just like @hydra.main would
    cfg: DictConfig = compose(config_name=pipeline)
    print("Loaded config:\n", OmegaConf.to_yaml(cfg, resolve=True))

    # 3) Run your pipeline
    pipeline = MediaPipeline(cfg, media_cache_dir=media_cache_dir)
    result = pipeline.run(
        video_path=video_path,
        img_dir=img_dir,
        mask_path=mask_path,
        frame_idxs=frame_idxs,
        jumble_map=jumble_map,
    )
    print("Pipeline output:", result)
    return result
    

if __name__ == "__main__":
    """
    python3 pipeline.py \
        --video_dir data/videos/keyframe-video/fps-1/ \
        --question_yaml data/questions/yamls/098.yaml \
        --img_dir data/rgb-frames/ \
        --mask_dir data/segmentation-masks \
        --media_cache_dir tmp/media_cache/ \
        --pipeline media_pipeline/collage_pipeline
    """
    import argparse
    parser = argparse.ArgumentParser(description="Generate visual prompts.")
    parser.add_argument("--question_yaml", type=str, required=True, help="Path to the question YAML file.")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing the RGB frames.")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory containing the mask files.")
    parser.add_argument("--video_dir", type=str, help="Path to the video directory.")
    parser.add_argument("--pipeline", type=str, default="separate_pipeline", help="Name of the pipeline to run.")
    parser.add_argument("--media_cache_dir", type=str, default="/tmp/media_cache", help="Directory to cache media files.")
    args = parser.parse_args()
    main(
        video_dir=args.video_dir,
        question_yaml=args.question_yaml,
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        pipeline=args.pipeline,
        media_cache_dir=args.media_cache_dir  # Default cache directory
    )