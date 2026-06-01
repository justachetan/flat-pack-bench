import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from typing import Dict, Any, Literal
import os
import os.path as osp
import copy
import numpy as np
import torch
import mediapy
import imageio.v2 as iio
from torchvision.utils import make_grid
from PIL.Image import Image
from PIL import Image as PILImage

from src.eval.prompts.media.components.base import MediaComponent


class CollageComponent(MediaComponent):
    
    def __init__(
        self,
        border_size: int = 10,
        border_color: float = 0,
        prompt_pos: Literal["top", "bottom", "left", "right"] = "bottom",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.prompt_pos = prompt_pos
        self.border_size = border_size
        self.border_color = border_color

    def get_build_params(self, media: Dict[str, Any], input_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get the parameters used for building the collage component.
        This can be used to generate a cache key or for logging.
        """
        params = {
            "video": media["video"],
            "visual_prompts": media["visual_prompts"],
            "vp_frame_idxs": media["vp_frame_idxs"],
        }
        
        return params
    
    def get_component_params(self):
        return {
            "border_size": self.border_size,
            "border_color": self.border_color,
            "prompt_pos": self.prompt_pos,
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
            media_cache_dir (str): Directory to cache the processed media.
            override_cache (bool): Whether to override existing cached media.
        Returns:
            dict: A dictionary containing the processed media.
        """
        video_fn = media["video"]  # ndarray (T, H, W, C)
        vp_fn_dict = media.get("visual_prompts", {})
        vp_idxs = media.get("vp_frame_idxs", [])

        # convert PIL or numpy image to CHW tensor
        def to_tensor(img_arr):
            if isinstance(img_arr, Image):
                arr = np.array(img_arr)
            else:
                arr = img_arr
            return torch.from_numpy(arr.transpose(2, 0, 1))

        cache_fn = self.get_cache_key(media)
        if osp.exists(osp.join(media_cache_dir, f"{cache_fn}.mp4")) and (not override_cache):
            collage_media = copy.deepcopy(media)
            collage_media["video"] = f"{cache_fn}.mp4"
            return collage_media
        
        video = mediapy.read_video(osp.join(media_cache_dir, video_fn))
        vp_dict = {
            idx: iio.imread(
                osp.join(media_cache_dir, vp_fn_dict[idx])
            ) for idx in vp_fn_dict 
        }

        # Preload and resize prompt images to match video frame dimensions
        # First element for dimension lookup
        sample_frame = video[0]
        H_v, W_v = sample_frame.shape[:2]
        resized_prompts = []
        for idx in vp_idxs:
            if idx not in vp_fn_dict:
                raise KeyError(f"visual_prompts missing index {idx}")
            arr = vp_dict[idx]
            # Convert array to PIL, resize, back to numpy
            pil = PILImage.fromarray(arr)
            pil_resized = pil.resize((W_v, H_v))
            resized_arr = np.array(pil_resized)
            resized_prompts.append(to_tensor(resized_arr))

        new_frames = []
        for frame in video:
            frame_tensor = to_tensor(frame)
            # Assemble prompt tensors and frame tensor in correct order
            if self.prompt_pos == "left":
                imgs = resized_prompts + [frame_tensor]
            elif self.prompt_pos == "right":
                imgs = [frame_tensor] + resized_prompts
            elif self.prompt_pos == "top":
                imgs = resized_prompts + [frame_tensor]
            else:  # bottom
                imgs = [frame_tensor] + resized_prompts

            # Determine nrow: horizontal for left/right, vertical for top/bottom
            if self.prompt_pos in ("left", "right"):
                nrow = len(imgs)
            else:
                nrow = 1

            # Single make_grid call
            grid = make_grid(imgs, nrow=nrow,
                             padding=self.border_size,
                             pad_value=self.border_color)
            # back to numpy HWC
            arr = grid.numpy().transpose(1, 2, 0)
            new_frames.append(arr)

        mediapy.write_video(
            osp.join(media_cache_dir, f"{cache_fn}.mp4"),
            new_frames,
            fps=1
        )

        collage_media = copy.deepcopy(media)
        collage_media["video"] = f"{cache_fn}.mp4"
        return collage_media
    
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
    
    
    for pos in ["top", "bottom", "left", "right"]:
        collage_component = CollageComponent(
            media_cache_dir=media_cache_dir,
            border_size=10,
            border_color=0,
            prompt_pos=pos,
        )
        collage_media = collage_component.build(media)
        print(f"after collage ({pos}):")
        print(collage_media)
        
        
if __name__ == "__main__":
    """
    python3 collage.py \
        --video_path data/videos/keyframe-video/fps-1/Chair/vedbo/NdkuJ9cwOuE/NdkuJ9cwOuE.mp4 \
        --question_yaml data/questions/yamls/098.yaml \
        --img_dir data/rgb-frames/ \
        --mask_dir data/segmentation-masks
    """
    import argparse
    parser = argparse.ArgumentParser(description="Generate visual prompts.")
    parser.add_argument("--question_yaml", type=str, required=True, help="Path to the question YAML file.")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing the RGB frames.")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory containing the mask files.")
    parser.add_argument("--video_path", type=str, help="Path to the video file.")
    args = parser.parse_args()
    main(args.video_path, args.question_yaml, args.img_dir, args.mask_dir)