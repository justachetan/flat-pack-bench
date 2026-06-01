import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from typing import Tuple, Literal, List, Any, Dict, Union
import json
import copy
import yaml
import os
import os.path as osp

import imageio.v2 as iio
import mediapy
import random
import numpy as np
from PIL import Image

from pytorch_lightning import seed_everything

from src.eval.prompts.media.components.base import MediaComponent
from src.eval.prompts.media.components.som_visualization import generate_som_prompt_image

class VisualPrompt(MediaComponent):
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (480, 640),
        cmap: str = "tab20",
        alpha: float = 0.1,
        anno_mode: List[Literal["Mask", "Box", "Mark"]] = ["Mask", "Mark"],
        area_threshold: float = 0.01,
        label_mode: Literal["1", "A"] = "1",
        color_part_by_id: bool = True,
        edgewidth: int = 2,
        high_contrast_colors: bool = False,
        high_contrast_colors_n_spls: int = 10000,
        high_contrast_ref_spls: int = 1000,
        high_contrast_colors_method: str = 'lab',
        seed: int = 42,
        use_expanded_cache_key: bool = False,
        font_size: int = 18,
        **kwargs
    ):
        """Initialize the visual prompt.

        Args:
            resolution (Tuple[int, int], optional): The resolution of the output image (H x W). 
                Defaults to (480, 640).
            cmap (str, optional): The colormap to use for visualization. Defaults to "tab20".
            alpha (float, optional): The transparency level for overlays. Defaults to 0.1.
            anno_mode (List[Literal["Mask", "Box", "Mark"]], optional): The annotation mode to use. Defaults to ["Mask", "Box", "Mark"].
            area_threshold (float, optional): The area threshold for small objects. Defaults to 0.01.
            label_mode (Literal["1", "A"], optional): The label mode to use. Defaults to "1".
            color_part_by_id (bool, optional): Whether to color parts by their ID. Defaults to True.
            edgewidth (int, optional): Width of the edge to draw around masks. Defaults to 2.
            high_contrast_colors (bool, optional): If True, will sample high contrast colors for each mask. Will ignore colors and edge_colors.
                Defaults to False.
            high_contrast_colors_n_spls (int, optional): Number of samples to draw for high contrast colors.
                Default is 10000.
            high_contrast_ref_spls (int, optional): Number of reference samples to draw for high contrast colors.
                Default is 1000.
            high_contrast_colors_method (Literal['lab', 'wcag'], optional): Method to use for high contrast color sampling.
                'lab' uses CIELAB distance, 'wcag' uses WCAG contrast ratio.
                Default is 'lab'.
            seed (int, optional): Random seed for color sampling. Defaults to 42.
            use_expanded_cache_key (bool, optional): Whether to use an expanded cache key.
                TODO: phase this out later. Defaults to False.
            font_size (int, optional): Font size for labels. Defaults to 18.
        """
        super().__init__(**kwargs)
        self.resolution = resolution
        self.cmap = cmap
        self.alpha = alpha
        self.anno_mode = anno_mode
        self.area_threshold = area_threshold
        self.label_mode = label_mode
        self.color_part_by_id = color_part_by_id
        self.edgewidth = edgewidth
        self.high_contrast_colors = high_contrast_colors
        self.high_contrast_colors_n_spls = high_contrast_colors_n_spls
        self.high_contrast_ref_spls = high_contrast_ref_spls
        self.high_contrast_colors_method = high_contrast_colors_method
        self.font_size = font_size
        self.seed = seed
        self.use_expanded_cache_key = use_expanded_cache_key
        
        
    def get_build_params(self, media: Dict[str, Any], input_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get the parameters used for building the visual prompt.
        This can be used to generate a cache key or for logging.
        """
        params = {
            "img_dir": input_params["img_dir"],
            "mask_path": input_params["mask_path"],
            "frame_idxs": input_params["frame_idxs"],
            "colors": input_params.get("colors", None),
            "edge_colors": input_params.get("edge_colors", None),
            "jumble_map": input_params.get("jumble_map", None),
        }
        params.update(media)
        return params
        
    def get_component_params(self) -> Dict[str, Any]:
        """
        Get the parameters of the visual prompt component.
        This can be used to generate a cache key or for logging.
        """
        component_params = {
            "resolution": self.resolution,
            "cmap": self.cmap,
            "alpha": self.alpha,
            "anno_mode": self.anno_mode,
            "area_threshold": self.area_threshold,
            "label_mode": self.label_mode,
            "color_part_by_id": self.color_part_by_id,
            # NOTE: commenting out high contrast params to avoid cache explosion
            # "high_contrast_colors": self.high_contrast_colors,
            # "high_contrast_colors_n_spls": self.high_contrast_colors_n_spls,
            # "high_contrast_ref_spls": self.high_contrast_ref_spls,
            # "high_contrast_colors_method": self.high_contrast_colors_method,
            # "edgewidth": self.edgewidth,
        }
        if self.use_expanded_cache_key:
            component_params.update({
                "seed": self.seed,
                "edgewidth": self.edgewidth,
                "font_size": self.font_size,
                "high_contrast_colors": self.high_contrast_colors,
                "high_contrast_colors_n_spls": self.high_contrast_colors_n_spls,
                "high_contrast_ref_spls": self.high_contrast_ref_spls,
                "high_contrast_colors_method": self.high_contrast_colors_method,
            })
        return component_params

        
    def get_cache_key(self, media: Dict[str, Any], input_params: Dict[str, Any] = None) -> str:
        """
        Generate a cache file name based on parameters and build arguments.
        This is used to avoid recomputing the same media component.
        """
        return super().get_cache_key(media, input_params)
        
    def build(
        self,
        img_dir: str,
        mask_path: str,
        media_cache_dir: str,
        frame_idxs: Union[List[int], int],
        colors: Any = None,
        edge_colors: Any = None,
        media: Dict[str, Any] = None,
        jumble_map: Dict[str, str] = None,
        override_cache: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate visual prompt images for a list of frame indices.

        Args:
            img_dir (str): Directory containing the RGB frames.
            mask_path (str): Path to the mask file.
                frame_idx
                    |-> part_id
                        |-> RLE mask
            frame_idxs (List[int]): List of frame indices to use.
            colors (Any, optional): Colors to use for the parts. Defaults to None.
            edge_colors (Any, optional): Edge colors to use for the parts. Defaults to None.
            media (Dict[str, Any], optional): Existing media dictionary to update. Defaults to None.
            jumble_map (Dict[str, str], optional): Mapping of part IDs to new IDs. Defaults to None.
                if provided, will remap the part IDs in the mask, and only do this for the last frame index
                in the provided `frame_idxs`.
            override_cache (bool, optional): Whether to override cached media. Defaults to False.
        Returns:
            Dict[str, Any]: Dictionary with key 'visual_prompts' mapping to a list of prompt images.
        """
        
        if isinstance(frame_idxs, int):
            frame_idxs = [frame_idxs]
        
        # import ipdb; ipdb.set_trace()
        # Load all masks once
        with open(mask_path) as f:
            masks = json.load(f)["manual"]
            
        visual_prompts = dict()
        for idx in frame_idxs:
            
            input_params = {
                "img_dir": img_dir,
                "mask_path": mask_path,
                "frame_idxs": idx,
                "colors": colors,
                "edge_colors": edge_colors,
                "jumble_map": jumble_map,
            }
            
            # NOTE: the first condition is not needed as usually 
            #       questions with a single frame don't have a 
            #       jumble map, but just adding it for redundancy
            # NOTE: this condition below ensures that frames in tracking
            #       question always end up with unique IDs. As only the
            #       second frame retains the jumble map, ensuring that 
            #       its name does not match any other question image 
            #       where it was the only frame or was used as the first image
            if len(frame_idxs) > 1 and idx != frame_idxs[-1]:
                input_params.pop("jumble_map", None)
            cache_fn = self.get_cache_key(
                media,
                input_params=input_params
            )
            if osp.exists(osp.join(media_cache_dir, f"{cache_fn}.jpg")) and (not override_cache):
                visual_prompts[idx] = f"{cache_fn}.jpg"
                continue
            
            
            # Read the RGB frame
            img = iio.imread(osp.join(img_dir, f"{idx}.jpg"))
            mask = masks.get(str(idx), {})
            
            if jumble_map is not None and idx == frame_idxs[-1]:
                new_mask = dict()
                for k, v in jumble_map.items():
                    if k in mask:
                        new_mask[v] = copy.deepcopy(mask[k])
                mask = new_mask
                    
            # Generate the prompt image
            if self.seed is not None:
                np.random.seed(self.seed)
                random.seed(self.seed)
                seed_everything(self.seed)

            vp_img = generate_som_prompt_image(
                img=img,
                masks=mask,
                cmap=self.cmap,
                alpha=self.alpha,
                anno_mode=self.anno_mode,
                area_threshold=self.area_threshold,
                label_mode=self.label_mode,
                color_by_part_id=self.color_part_by_id,
                colors=colors,
                edge_colors=edge_colors,
                high_contrast_colors=self.high_contrast_colors,
                high_contrast_colors_n_spls=self.high_contrast_colors_n_spls,
                high_contrast_ref_spls=self.high_contrast_ref_spls,
                high_contrast_colors_method=self.high_contrast_colors_method,
                edgewidth=self.edgewidth,
                font_size=self.font_size,
            )
            
            # TODO: allow using cached colormaps in input
            colormap = None
            if self.high_contrast_colors:
                vp_img, colormap = vp_img

            # Resize if needed
            if self.resolution and self.resolution != np.array(vp_img).shape[:2]:
                vp_img = vp_img.resize(
                    (self.resolution[1], self.resolution[0]),
                )
            vp_img.save(
                osp.join(media_cache_dir, f"{cache_fn}.jpg"),
            )
            with open(osp.join(media_cache_dir, f"{cache_fn}_colormap.json"), 'w') as f:
                json.dump(colormap, f)

            visual_prompts[idx] = f"{cache_fn}.jpg"
        
        if media is None:
            media = dict()
        media.update({
            "visual_prompts": visual_prompts,
            "vp_frame_idxs": frame_idxs
        })
        return media
    
def main(question_yaml: str, img_dir: str, mask_dir: str):
    from src.eval.prompts.templates.questions.convert_yaml_to_json import convert_yaml_to_json
    
    media_cache_dir = osp.join(root, "tmp", "media_cache")
    os.makedirs(media_cache_dir, exist_ok=True)
    
    question_json = convert_yaml_to_json(question_yaml)
    category = question_json["vid_category"]
    name = question_json["furniture_name"]
    vid = question_json["video_id"]
    frame_idxs = question_json["frame_idx"]
    # jumble_map = question_json.get("jumble_map", None)
    with open(question_yaml) as yaml_f:
        jumble_map = yaml.safe_load(yaml_f).get("jumble_map", None)
    
    resolution = (480, 640)  # H x W
    img_dir = osp.join(img_dir, category, name, vid)
    mask_path = osp.join(mask_dir, category, name, vid, f"{vid}.json")
    colors = None
    edge_colors = None
    media = dict()

    visual_prompt_component = VisualPrompt(
        media_cache_dir=media_cache_dir,
        resolution=resolution,
    )
    # import ipdb; ipdb.set_trace()
    visual_prompts = visual_prompt_component.build(
        media_cache_dir=media_cache_dir,
        img_dir=img_dir,
        mask_path=mask_path,
        frame_idxs=frame_idxs,
        colors = colors,
        edge_colors=edge_colors,
        jumble_map=jumble_map,
        media=media,
    )
    print(visual_prompts)

if __name__ == "__main__":
    """
    python3 visual_prompt.py --question_yaml path/to/question.yaml --img_dir path/to/img_dir --mask_dir path/to/mask_dir
    ex. python3 visual_prompt.py \
        --question_yaml data/questions/yamls/211.yaml \
        --img_dir data/rgb-frames/ \
        --mask_dir data/segmentation-masks
    """
    import argparse
    parser = argparse.ArgumentParser(description="Generate visual prompts.")
    parser.add_argument("--question_yaml", type=str, required=True, help="Path to the question YAML file.")
    parser.add_argument("--img_dir", type=str, required=True, help="Directory containing the RGB frames.")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory containing the mask files.")
    
    args = parser.parse_args()
    main(args.question_yaml, args.img_dir, args.mask_dir)