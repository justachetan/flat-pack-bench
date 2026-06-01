import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from typing import Literal, Optional, Tuple, Union, Dict, List, Any


import os
import re
import os.path as osp
import uuid
import yaml
import json
import tempfile

import numpy as np
import imageio.v2 as iio

from PIL import Image
from pycocotools import mask as mask_utils

from src.eval.prompts.media.components.som_visualization import generate_som_prompt_image 
from src.tva.models import forward, model_classes
from src.tva.media.prompts.generate_api_prompt import api_desc
from src.tva.utils.parser import (
    override_from_method_params,
    no_overrides
)
from src.tva.media.video_segment import encode_mask
from src.tva.utils.common import log_event, logger


def extract_json_object(s: str):
    """
    Extracts a JSON object that contains at least the 'answer' key.
    Handles optional keys like 'explanation' and ignores extra text
    before/after the JSON.
    """
    # This regex captures the first {...} block that includes "answer"
    match = re.search(r'\{[\s\S]*?[\'"]answer[\'"][\s\S]*?\}', s)
    if not match:
        raise ValueError("No JSON object with 'answer' found.")

    json_str = match.group(0)

    # Try to clean common trailing commas (like JSON fragments)
    json_str = re.sub(r',\s*([\]}])', r'\1', json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}\nExtracted text:\n{json_str}")

@api_desc(
    description="""
        A class to handle images and their associated masks for creating visual
        prompts for Vision-Language Models (VLMs).
        
        Attributes
        ----------
            input_image (Union[str, Image.Image, np.ndarray]): Path to the image file, 
                a PIL Image object, or a numpy array representing the image.
            masks (Dict[str, Any]): A dictionary of masks associated with the image.
                Each key consists of a part ID and the value is either 
                a numpy array or a RLE-encoded binary mask.
            overlaid_input_image (str): Path to an image where binary part masks are 
                overlaid on the original image. Each part mask is shown in a distinct 
                color and labeled with a unique part ID.
            height (int): The height of the image.
            width (int): The width of the image.  
        
        Methods
        -------
            get_size() -> Tuple[int, int]:
                Returns the height and width of the image and mask.
            get_part_visibility(part_id: str):
                Returns the visibility of a part in the image given its part ID.
            vlm_query(
                query: str
            ) -> str:
                Returns the answer to a basic question (`query`) about the image and masks using a Vision-Language Model (VLM).
            check_part_connectivity(
                part_id1: str,
                part_id2: str
            ) -> bool:
                Checks if two parts are connected in the image using a VLM.
                  
    """,
    export=True
)
class ImagePatch(object):

    PROMPT_TEMPLATES = osp.abspath(osp.join(osp.dirname(__file__), "prompts"))
    
    @api_desc(
        description="""
        Initialize ImagePatch with image and mask file paths.

        Parameters
        ----------
            input_image (Union[str, Image.Image, np.ndarray]): Path to the image file, 
                a PIL Image object, or a numpy array representing the image.
            masks (Dict[str, Any]): Dictionary of masks for the image.
                Each key consists of a part ID and the value is either 
                a numpy array or a RLE-encoded binary mask.
        """,
        export=True,
        include_code=False,
        display_signature="(self,input_image: Union[str, Image.Image, np.ndarray], masks: Dict[str, Any])"
    )
    def __init__(
        self, 
        input_image: Union[str, Image.Image, np.ndarray],
        masks: Dict[str, Any],
        cache_dir: Optional[str] = None,
        method_params: Dict[str, Any]=None,
        **kwargs
    ):
        """Initialize ImagePatch with image and mask file paths.

        Args:
            input_image (Union[str, Image.Image, np.ndarray]): The input image.
            masks (Dict[str, Any]): Dictionary of masks for the image.
            cache_dir (Optional[str]): Directory to cache temporary files. Defaults to None.
            method_params (Dict[str, Any]): Dictionary of method parameters to override defaults. Defaults to None.
        """
        self.input_image = input_image if isinstance(input_image, str) else None
        self.masks = masks
        self.cache_dir = cache_dir if cache_dir is not None else tempfile.gettempdir()
        os.makedirs(self.cache_dir, exist_ok=True)
        # if not isinstance(list(self.masks.values())[0], np.ndarray):
        #     for k in self.masks.keys():
        #         self.masks[k] = mask_utils.decode(self.masks[k])
        self.image = None
        if isinstance(input_image, np.ndarray):
            self.image = Image.fromarray(input_image).convert("RGB")
            self.input_image = self._save_img_as_tempfile(self.image)
        elif isinstance(input_image, Image.Image):
            self.image = input_image.convert("RGB")
            self.input_image = self._save_img_as_tempfile(self.image)
        elif isinstance(input_image, str):
            self.image = Image.open(input_image).convert("RGB")


        self.height, self.width = np.array(self.image).shape[:2]
        self.method_params = method_params or {}

        # encode masks if they are numpy arrays
        # as subsequent methods expect RLE-encoded masks
        for k, v in self.masks.items():
            if isinstance(v, np.ndarray):
                self.masks[k] = encode_mask(v)
        
        # TODO: in case of high-contrast colors, need a way to save 
        # the color map used for visualization
        # import ipdb; ipdb.set_trace()
        if self.masks is not None and len(self.masks) > 0:
            self.overlaid_image = generate_som_prompt_image(
                img=np.array(self.image),
                masks=self.masks,
                cmap=kwargs.get("cmap", "tab20"),
                alpha=kwargs.get("alpha", 0.1),
                anno_mode=kwargs.get("anno_mode", ["Mask", "Mark"]),
                area_threshold=kwargs.get("area_threshold", 10),
                label_mode=kwargs.get("label_mode", "1"),
                color_by_part_id=kwargs.get("color_by_part_id", False),
                high_contrast_colors=kwargs.get("high_contrast_colors", True),
                high_contrast_colors_n_spls=kwargs.get("high_contrast_colors_n_spls", 5000),
                high_contrast_ref_spls=kwargs.get("high_contrast_ref_spls", 100),
                high_contrast_colors_method=kwargs.get("high_contrast_colors_method", "lab"),
            )
        else:
            self.overlaid_image = self.image
        if isinstance(self.overlaid_image, tuple):
            self.overlaid_image, self.part_id_to_color_map = self.overlaid_image
        self.overlaid_input_image = self._save_img_as_tempfile(self.overlaid_image)

        log_event(
            stage="image_patch", 
            event="init", 
            msg=f"ImagePatch(input_image={osp.basename(self.input_image)}, overlaid_img_fn={osp.basename(self.overlaid_input_image)}, num_masks={len(self.masks)})", meta={
            "input_image": self.input_image,
            "num_masks": len(self.masks),
            "part_ids": list(self.masks.keys()),
            "height": self.height,
            "width": self.width,
            "overlaid_input_image": self.overlaid_input_image,
        }, file_path=self.overlaid_input_image)
    
    def _overlay_masks_on_image(self) -> Image.Image:
        """Overlay masks on the image.

        Returns
        -------
            Image.Image: The image with masks overlaid.
        """
        return self.overlaid_image
    
    @api_desc(
        description="""
        Returns the height and width of the image and mask.

        Returns
        -------
            Tuple[int, int]: The height and width of the image and mask.
        """,
        include_code=True,
        parent_blurb="Returns the height and width of the image and mask.",
        export=True,
    )
    def get_size(self) -> Tuple[int, int]:
        """Get the size of the image and mask.

        Returns:
            Tuple[int, int]: The height and width of the image and mask.
        """
        return self.height, self.width
    
    def _save_img_as_tempfile(self, img: Image.Image) -> str:
        """Save the image as a temporary file.

        Args:
            img (Image.Image): The image to save.
        Returns:
            str: The path to the temporary file.
        """

        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=self.cache_dir)
        img.save(temp_file.name)
        # logger.info(f"saved overlaid image to temporary file: {temp_file.name}")
        return temp_file.name

    @override_from_method_params
    @api_desc(
        description="""
        Returns the visibility of a part in the image given its part ID.

        Parameters
        ----------
            part_id (str): The part ID, this is an int in string format.
        Returns
        -------
            bool: True if the part is visible, False otherwise.
            
        Examples
        --------
        # Is part "0" visible in the image?
        >>> def execute_code(input_image, masks):
        >>>     image_patch = ImagePatch(input_image=input_image, masks=masks)
        >>>     is_visible = image_patch.get_part_visibility(part_id="0")
        >>>     return is_visible
        """,
        export=True,
        body="""
            return get_part_visibility(part_id)
        """,
        parent_blurb="Returns the visibility of a part in the image given its part ID.",
        display_signature="(self, part_id: str) -> bool"
    )
    def get_part_visibility(
        self,
        part_id: str,
        visibility_threshold: float = 0.,
    ) -> float:
        """Get the visibility of a part in the image.

        Args:
            part_id (str): The part ID.
            visibility_threshold (float): The threshold to consider a part as visible. Defaults to 0.

        Returns:
            float: The visibility of the part in the image (0 to 1).
        """
        if part_id not in self.masks:
            raise ValueError(f"Part ID {part_id} not found in masks.")
        
        mask = self.masks[part_id]
        if not isinstance(mask, np.ndarray):
            mask = mask_utils.decode(mask)
        
        visible_area = np.sum(mask)
        total_area = self.height * self.width
        visibility = visible_area / total_area
        
        return visibility > visibility_threshold
    
    @override_from_method_params
    @api_desc(
        description="""
        Returns the answer to a basic question (`query`) about the image and masks using a Vision-Language Model (VLM).
        The questions are about basic perception, and are not meant to be used for complex reasoning or external
        knowledge.
        
        Parameters
        ----------
            query (str): The query string.
        
        Returns
        -------
            str: The response from the VLM model.
            
        Examples
        --------
        # What is shown in the image?
        >>> def execute_code(input_image, masks):
        >>>     image_patch = ImagePatch(input_image=input_image, masks=masks)
        >>>     response = image_patch.vlm_query(
        >>>         query="What is shown in the image?"
        >>>     )
        >>>     return response
        """,
        body="""
            return vlm_query(query)
        """,
        export=True,
        parent_blurb="Returns the answer to a basic question (`query`) about the image and masks using a Vision-Language Model (VLM).",
        display_signature="(self, query: str) -> str"
    )
    def vlm_query(
        self,
        *args,
        query: str = None,
        vlm_model: Literal[
            "gpt4v", 
            "gemini",
            "qwen25"
        ]="gemini",
        prompt_template: Literal["default", "contact"] = "default",
        model_init_args: Dict[str, Any]=None,
        msg_params: Dict[str, Any]=None,
        print_msgs: bool=False,
        **kwargs,
    ):
        """Perform VLM query on the image and mask.

        Args:
            query (str): The query string.
            vlm_model (Literal): The VLM model to use. Defaults to "gemini".
                Options are "gpt4v", "gemini", "qwen25".
            prompt_template (Literal): The prompt template to use. Defaults to "default".
                Options are "default", "detailed", "contact", or a full path to a custom YAML file.
            model_init_args (Dict[str, Any]): The initialization arguments for the VLM model.
                Defaults to None.
            msg_params (Dict[str, str]): Parameters to customize specific messages in the prompt template.
                Defaults to None.
                Structure: {
                    <tag_of_msg_to_customize>: {<param_name>: <param_value>, ...}, ...
                }
            print_msgs (bool): Whether to print the messages in the prompt. Defaults to False.
        kwargs: Additional arguments to pass to the VLM model.
        Returns:
            str: The response from the VLM model.
        """
        log_event(stage="image_patch", event="vlm_query", msg="VLM query started", meta={
            "query": query,
            "vlm_model": vlm_model,
            "prompt_template": prompt_template,
            "model_init_args": model_init_args,
            "msg_params": msg_params,
            "print_msgs": print_msgs,
        })
        prompt_template_fn = None
        if not osp.exists(prompt_template):
            if not osp.exists(
                osp.join(
                    self.PROMPT_TEMPLATES, f"{prompt_template}.yaml"
                    if prompt_template.endswith(".yaml") 
                    else f"{prompt_template}.yaml"
                )
            ):
                raise ValueError(f"Prompt template {prompt_template} not found in {self.PROMPT_TEMPLATES}. Please provide full path.")
            else:
                prompt_template_fn = osp.join(
                    self.PROMPT_TEMPLATES, 
                    f"{prompt_template}.yaml" if prompt_template.endswith(".yaml") else f"{prompt_template}.yaml"
                )    
        else:
            prompt_template_fn = prompt_template
        
        if query == "None" or query == "":
            query = "What is shown in the image?"
        
        with open(prompt_template_fn, "r") as f:
            prompt_json = yaml.safe_load(f)
        
        for i, msg in enumerate(prompt_json):
            
            if msg["type"] == "text":
                msg_content = msg["content"]

                if msg["tag"] == "question" and query is not None:
                    msg_content = query

                if msg_params is not None and msg["tag"] in msg_params:
                    msg_content = msg_content.format(**msg_params[msg["tag"]])
                
                prompt_json[i]["content"] = msg_content


            elif msg["type"] == "image":
                if msg["tag"] == "visual_prompt":
                    prompt_json[i]["content"] = self.overlaid_input_image
                elif msg["tag"] == "raw_image":
                    prompt_json[i]["content"] = self.input_image
                elif msg["tag"] in msg_params:
                    prompt_json[i]["content"] = msg_params[msg["tag"]]
                else:
                    raise ValueError(f"Unknown image tag {msg['tag']}. Supported tags are 'visual_prompt' and 'raw_image'.")
                if osp.exists(prompt_json[i]["content"]):
                    # For vllm_online, need to use file:// URI scheme with file paths
                    if "forward_pipeline" in model_init_args and model_init_args["forward_pipeline"] == "vllm_online":
                        prompt_json[i]["content"] = f"{prompt_json[i]['content']}"
            else:
                import ipdb; ipdb.set_trace()
                raise ValueError(f"Unknown message type {msg['type']}. Supported types are 'text' and 'image'.")
        if print_msgs:
            # logger.info(json.dumps(prompt_json, indent=2))
            vlm_prompt_fn = osp.join(self.cache_dir, f"vlm_prompt_{uuid.uuid4().hex}.json")
            with open(vlm_prompt_fn, "w") as f:
                f.write(json.dumps(prompt_json, indent=2))
            log_event(stage="image_patch", event="vlm_query", msg="VLM prompt generated", file_path=vlm_prompt_fn)

        vlm_resp = self.forward(vlm_model, prompt_json, *args, init_args=model_init_args, **kwargs)
        resp_fn = osp.join(self.cache_dir, f"vlm_response_{uuid.uuid4().hex}.txt")
        with open(resp_fn, "w") as f:
            f.write(vlm_resp if isinstance(vlm_resp, str) else json.dumps(vlm_resp, indent=2))
        log_event(stage="image_patch", event="vlm_query", msg="VLM response received", meta={
            "vlm_raw_response": vlm_resp
        }, file_path=resp_fn)

        return vlm_resp


    @override_from_method_params
    @api_desc(
        description="""
        Check if two parts are connected in in the image using using the overlaid
        image (of the masks on the original image) with the help of a VLM.
        
        Parameters
        ----------
            part_id1 (str): The first part ID.
            part_id2 (str): The second part ID.
        Returns
        -------
            bool: True if the parts are connected, False otherwise.
            
        Examples
        --------
        # Are parts "0" and "1" connected in the image?
        >>> def execute_code(input_image, masks):
        >>>     image_patch = ImagePatch(input_image=input_image, masks=masks)
        >>>     response = image_patch.check_part_connectivity(
        >>>         part_id1="0",
        >>>         part_id2="1",
        >>>     )
        
        # Are parts "0" and "2" connected in the image?
        >>> def execute_code(input_image, masks):
        >>>     image_patch = ImagePatch(input_image=input_image, masks=masks)
        >>>     response = image_patch.check_part_connectivity(
        >>>         part_id1="0",
        >>>         part_id2="2",
        >>>     )
        >>>     return response
        """,
        export=False,
        body="""
            return check_part_connectivity(part_id1, part_id2, self.overlaid_input_image)
        """,
        parent_blurb="Checks if two parts are connected in the image using a VLM.",
        display_signature="(self, part_id1: str, part_id2: str) -> bool"
    )   
    def check_part_connectivity(
        self,
        part_id1: str,
        part_id2: str,
        vlm_model: Literal[
            "gpt4v", 
            "gemini",
            "qwen25"
        ]="gemini",
        model_init_args: Dict[str, Any]={
            "model_name": "gemini-2.5-pro",
            "generate_config": {"temperature": 0.0}
        },
        print_msgs: bool=False,
        **kwargs,
    ) -> bool:
        """Check if two parts are connected in the image using the overlaid image.

        Args:
            part_id1 (str): The first part ID.
            part_id2 (str): The second part ID.
        Returns:
            bool: True if the parts are connected, False otherwise.
        """
        if part_id1 not in self.masks or part_id2 not in self.masks:
            return False
        
        # NOTE/TODO: do we need to add mask dilation to allow the VLM to build a better
        # understanding of connectivity?
        
        text_params = {
            "question": {
                "query_part1": part_id1,
                "query_part2": part_id2,
            }
        }
        
        # logger.info(f"checking connectivity: part {part_id1} & part {part_id2}; VLM model {vlm_model}; prompt template: 'contact'")
        log_event(stage="image_patch", event="check_part_connectivity", msg=f"Checking connectivity between part {part_id1} and part {part_id2}", meta={
            "vlm_model": vlm_model,
            "model_init_args": model_init_args,
            "part_id1": part_id1,
            "part_id2": part_id2,
        })
        with no_overrides(self):
            vlm_resp = self.vlm_query(
                query=None,
                vlm_model=vlm_model,
                prompt_template="contact",
                model_init_args=model_init_args,
                msg_params=text_params,
                print_msgs=print_msgs,
            )
        
        if isinstance(vlm_resp, dict):
            vlm_resp = json.loads(vlm_resp["response"].strip().lower())
        else:
            vlm_resp = extract_json_object(vlm_resp.strip())
        # import ipdb; ipdb.set_trace()
        # logger.info(f"Extracted VLM response JSON: {vlm_resp}")
        log_event(stage="image_patch", event="check_part_connectivity", msg=f"VLM response received: {vlm_resp}", meta={
            "vlm_response": vlm_resp
        })
        assert "answer" in vlm_resp, f"VLM response does not contain 'answer' key: {vlm_resp}"
        assert vlm_resp["answer"].lower().strip() in ["yes", "no"], f"VLM response 'answer' is not 'yes' or 'no': {vlm_resp}"
        is_connected = vlm_resp["answer"].lower().strip() == "yes"
        log_event(stage="image_patch", event="check_part_connectivity", msg=f"Parts connectivity result: {is_connected}", meta={
            "is_connected": is_connected
        })
        return is_connected

    @override_from_method_params
    @api_desc(
        description="""
        Returns all pairs of connected parts in an image as a list of tuples.
        Uses the overlaid image (of the masks on the original image) to visualize
        the parts and outputs all connected part pairs with the help of a VLM.

        Returns
        -------
            List[Tuple[str, str]]: A list of tuples where each tuple contains two part IDs that are connected.

        Examples
        --------
        # Get all connected part pairs in the image.
        # Which pairs of parts are connected in the image?
        >>> def execute_code(input_image, masks):
        >>>     image_patch = ImagePatch(input_image=input_image, masks=masks)
        >>>     connected_pairs = image_patch.get_all_connected_part_pairs()
        >>>     return connected_pairs

        # Are parts "0" and "1" connected in the image?
        >>> def execute_code(input_image, masks):
        >>>     image_patch = ImagePatch(input_image=input_image, masks=masks)
        >>>     response = image_patch.get_all_connected_part_pairs()
        >>>     for pair in response:
        >>>         if pair == ("0", "1") or pair == ("1", "0"):
        >>>             return True
        >>>     return False

        # How many pairs of connected parts are there in the image?
        >>> def execute_code(input_image, masks):
        >>>     image_patch = ImagePatch(input_image=input_image, masks=masks)
        >>>     response = image_patch.get_all_connected_part_pairs()
        >>>     return len(response)
        """,
        export=True,
        body="""
            return get_all_connected_part_pairs(self.overlaid_input_image)
        """,
        parent_blurb="Returns all pairs of connected parts in an image as a list of tuples.",
        display_signature="(self) -> List[Tuple[str, str]]"
    )   
    def get_all_connected_part_pairs(
        self,
        vlm_model: Literal[
            "gpt4v", 
            "gemini",
            "qwen25"
        ]="gemini",
        model_init_args: Dict[str, Any]={
            "model_name": "gemini-2.5-pro",
            "generate_config": {"temperature": 0.0}
        },
        print_msgs: bool=False,
        **kwargs,
    ) -> List[Tuple[str, str]]:
        """Get all pairs of connected parts in the image.

        Returns:
            List[Tuple[str, str]]: A list of tuples where each tuple contains two part IDs that are connected.
        """
        connected_pairs = []
        part_ids = list(self.masks.keys())
        part_ids = [pid for pid in part_ids if self.get_part_visibility(pid, visibility_threshold=0.0)]

        if len(part_ids) < 2:
            log_event(stage="image_patch", event="get_all_connected_part_pairs", msg="Not enough parts to check connectivity", meta={
                "part_ids": part_ids,
            })
            return connected_pairs

        part_ids_str = ', '.join([f'"{pid}"' for pid in part_ids[:-1]]) + f', and "{part_ids[-1]}"'

        log_event(stage="image_patch", event="get_all_connected_part_pairs", msg=f"Getting all connected part pairs from parts {part_ids_str}", meta={
            "part_ids": part_ids,
            "vlm_model": vlm_model,
            "model_init_args": model_init_args,
        })

        with no_overrides(self):
            text_params = {
                "question": {
                    "part_list_str": part_ids_str
                }
            }
            vlm_resp = self.vlm_query(
                query=None,
                vlm_model=vlm_model,
                prompt_template="contact_all_pairs",
                model_init_args=model_init_args,
                msg_params=text_params,
                print_msgs=print_msgs,
            )

            if isinstance(vlm_resp, dict):
                vlm_resp = json.loads(vlm_resp["response"].strip().lower())
            else:
                vlm_resp = extract_json_object(vlm_resp.strip())

        connected_pairs = vlm_resp["answer"]

        if isinstance(connected_pairs, str):
            connected_pairs = json.loads(connected_pairs)

        if len(connected_pairs) > 0:
            assert all(isinstance(pair, list) and len(pair) == 2 for pair in connected_pairs), \
                f"VLM response 'answer' is not a list of pairs: {vlm_resp}"

        # remove duplicate edges
        connected_pairs = list(set([tuple(sorted(pair)) for pair in connected_pairs]))

        # remove hallucinated part pairs
        connected_pairs = [pair for pair in connected_pairs if pair[0] in part_ids and pair[1] in part_ids]
        return connected_pairs

    @override_from_method_params
    @api_desc(
        description="""
        Given a new image as input (without any masks) which contains some parts from the original image,
        finds the pairs of parts that are connected in the new image with the 
        help of a VLM.

        Parameters
        ----------
            new_input_image (Union[str, Image.Image, np.ndarray]): The new input image.
        Returns
        -------
            List[Tuple[str, str]]: A list of tuples where each tuple contains two part
                IDs that are connected in the new image.

        Examples
        --------
        # Are parts "0" and "1" connected in the new image?
        >>> def execute_code(input_image, masks, new_input_image):
        >>>     image_patch = ImagePatch(input_image=input_image, masks=masks)
        >>>     response = image_patch.track_and_find_all_connected_pairs(
        >>>         new_input_image=new_input_image,
        >>>     )
        >>>     for pair in response:
        >>>         if pair == ("0", "1") or pair == ("1", "0"):
        >>>             return True
        >>>     return False
        """,
        export=True,
        parent_blurb="""
            Finds the pairs of parts that are connected in the new input image.
        """,
        body="""
            return track_and_find_all_connected_pairs(self.overlaid_input_image, new_input_image)
        """,
        display_signature="(self, new_input_image: Union[str, Image.Image, np.ndarray]) -> List[Tuple[str, str]]"
    )
    def track_and_find_all_connected_pairs(
        self,
        new_input_image: Union[str, Image.Image, np.ndarray],
        vlm_model: Literal[
            "gpt4v", 
            "gemini",
            "qwen25"
        ]="gemini",
        model_init_args: Dict[str, Any]={
            "model_name": "gemini-2.5-pro",
            "generate_config": {"temperature": 0.0}
        },
        print_msgs: bool=False,
        **kwargs,
    ):
        """Track parts in a new image and find connected parts.

        Args:
            new_input_image (Union[str, Image.Image, np.ndarray]): The new input image.
            vlm_model (Literal): The VLM model to use. Defaults to "gemini".
                Options are "gpt4v", "gemini", "qwen25".
            model_init_args (Dict[str, Any]): The initialization arguments for the VLM model.
                Defaults to {
                    "model_name": "gemini-2.5-pro",
                    "generate_config": {"temperature": 0.0}
                }.
            print_msgs (bool): Whether to print the messages in the prompt. Defaults to False.
        kwargs: Additional arguments to pass to the VLM model.
        Returns:
            List[Tuple[str, str]]: A list of tuples where each tuple contains two part
                IDs that are connected in the new image.
        """
        connected_pairs = []
        part_ids = list(self.masks.keys())
        part_ids = [pid for pid in part_ids if self.get_part_visibility(pid, visibility_threshold=0.0)]

        if len(part_ids) < 2:
            log_event(stage="image_patch", event="track_and_find_all_connected_pairs", msg="Not enough parts to check connectivity", meta={
                "part_ids": part_ids,
            })
            return connected_pairs

        part_ids_str = ', '.join([f'"{pid}"' for pid in part_ids[:-1]]) + f', and "{part_ids[-1]}"'

        log_event(stage="image_patch", event="track_and_find_all_connected_pairs", msg="Tracking and finding connected parts in new image", meta={
            "vlm_model": vlm_model,
            "model_init_args": model_init_args,
            "print_msgs": print_msgs,
        })

        if not isinstance(new_input_image, str):
            # Save the new input image to a temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=self.cache_dir)
            if isinstance(new_input_image, np.ndarray):
                img = Image.fromarray(new_input_image).convert("RGB")
            elif isinstance(new_input_image, Image.Image):
                img = new_input_image.convert("RGB")
            img.save(temp_file.name)
            new_input_image = temp_file.name
            log_event(stage="image_patch", 
                      event="track_and_find_all_connected_pairs", 
                      msg=f"Saved new input image to temporary file: {new_input_image}", 
                      file_path=new_input_image
            )

        
        with no_overrides(self):
            vlm_resp = self.vlm_query(
                query=None,
                vlm_model=vlm_model,
                prompt_template="track_and_find_contact_all_pairs",
                model_init_args=model_init_args,
                msg_params={
                    "visual_prompt_2": new_input_image,
                    "question": {
                        "part_list_str": part_ids_str
                    }
                },
                print_msgs=print_msgs,
            )
            
        if isinstance(vlm_resp, dict):
            vlm_resp = json.loads(vlm_resp["response"].strip().lower())
        else:
            vlm_resp = extract_json_object(vlm_resp.strip())
        log_event(stage="image_patch", event="track_and_find_all_connected_pairs", msg=f"VLM response received: {vlm_resp}", meta={
            "vlm_response": vlm_resp
        })

        connected_pairs = vlm_resp["answer"]
        if isinstance(connected_pairs, str):
            connected_pairs = json.loads(connected_pairs)


        if len(connected_pairs) > 0:
            assert all(isinstance(pair, list) and len(pair) == 2 for pair in connected_pairs), \
                f"VLM response 'answer' is not a list of pairs: {vlm_resp}"

        # remove duplicate edges
        connected_pairs = list(set([tuple(sorted(pair)) for pair in connected_pairs]))

        # remove hallucinated part pairs
        connected_pairs = [pair for pair in connected_pairs if pair[0] in part_ids and pair[1] in part_ids]
        return connected_pairs

    
    @override_from_method_params
    @api_desc(
        description="""
        Given another ImagePatch instance of the same image with a different set of part masks, 
        finds all pairs of parts between the two ImagePatch instances that have overlapping masks based on a 
        specified IoU threshold.
        
        Parameters
        ----------
            img_patch (ImagePatch): Another ImagePatch instance to compare with.
            iou_threshold (float): The Intersection over Union (IoU) threshold to consider masks
                as overlapping. Defaults to 0.5.
                
        Returns
        -------
            List[Tuple[str, str]]: A list of tuples where each tuple contains two part IDs
                (from self and img_patch) that have overlapping masks.

        Examples
        --------
        # Which part in the new masks overlaps with part "0" in the original image?
        >>> def execute_code(input_image, masks_a, masks_b):
        >>>     image_patch_a = ImagePatch(input_image=input_image, masks=masks_a)
        >>>     image_patch_b = ImagePatch(input_image=input_image, masks=masks_b)
        >>>     overlapping_pairs = image_patch_a.find_overlapping_masks(
        >>>         img_patch=image_patch_b,
        >>>         iou_threshold=0.5,
        >>>     )
        >>>     return overlapping_pairs
        """,
        export=True,
        body="""
            return find_overlapping_masks(img_patch, iou_threshold)
        """,
        parent_blurb="""
            Given another ImagePatch instance of the same image with a different set of part masks, 
            finds all pairs of parts between the two ImagePatch instances that have overlapping masks based on a 
            specified IoU threshold.
        """,
        display_signature="(self, img_patch: 'ImagePatch', iou_threshold: float = 0.5) -> List[Tuple[str, str]]"
    )
    def find_overlapping_masks(
        self,
        img_patch: 'ImagePatch',
        iou_threshold: float = 0.5,
    ) -> List[Tuple[str, str]]:
        
        """Find overlapping masks between two ImagePatch instances.

        Args:
            img_patch (ImagePatch): Another ImagePatch instance to compare with.
            iou_threshold (float): The Intersection over Union (IoU) threshold to consider masks as overlapping. Defaults to 0.5.

        Returns:
            List[Tuple[str, str]]: A list of tuples where each tuple contains two part IDs
                (from self and img_patch) that have overlapping masks.
        """
        # import ipdb; ipdb.set_trace()
        overlapping_pairs = []
        for part_id1, mask1 in self.masks.items():
            mask1_enc = None
            if isinstance(mask1, np.ndarray):
                mask1_enc = encode_mask(mask1)
            else:
                mask1_enc = mask1
            for part_id2, mask2 in img_patch.masks.items():
                mask2_enc = None
                if isinstance(mask2, np.ndarray):
                    mask2_enc = encode_mask(mask2)
                else:
                    mask2_enc = mask2
                iou = float(mask_utils.iou([mask1_enc], [mask2_enc], [0])[0, 0])
                log_event(stage="image_patch", event="find_overlapping_masks", msg=f"Computed IoU between part {part_id1} and part {part_id2}: {iou}", meta={
                    "part_id1": part_id1,
                    "part_id2": part_id2,
                    "iou": iou,
                })
                if iou >= iou_threshold:
                    overlapping_pairs.append((part_id1, part_id2))
        log_event(stage="image_patch", event="find_overlapping_masks", msg=f"Found {len(overlapping_pairs)} overlapping part pairs ({overlapping_pairs}) with IoU >= {iou_threshold}", meta={
            "num_overlapping_pairs": len(overlapping_pairs),
            "iou_threshold": iou_threshold,
            "overlapping_pairs": overlapping_pairs,
        })
        return overlapping_pairs



    @override_from_method_params
    @api_desc(
        description="""
        Returns the answer to a basic question (`query`) about the image and masks using an LLM.
        
        The questions are about basic string matching, string parsing, and simple arithmetic.
        The questions are not meant to be used for complex reasoning or external knowledge.
        
        Parameters
        ----------
            query (str): The query string.
        Returns
        -------
            str: The response from the LLM.
            
        Examples
        --------
        # Which of the options matches the list of parts ["0", "1"] in the image?
        # Options:
        # A. Part 1 and Part 0
        # B. Part 0 and Part 2
        # C. Part 1 and Part 2
        # D. Part 1 and Part 3
        >>> def execute_code(input_image, masks):
        >>>     image_patch = ImagePatch(input_image=input_image, masks=masks)
        >>>     response = image_patch.simple_query(
        >>>         query="Which of the options matches the list of parts ['0', '1'] in the image?\\n Options:\\n A. Part 1 and Part 0\\n B. Part 0 and Part 2\\n C. Part 1 and Part 2\\n D. Part 1 and Part 3"
        >>>     )
        >>>     return response # should be "A"
        """,
        export=False, # was hurting performance in toy example (032.yaml)
        body="""
            return simple_query(query)
        """,
        display_signature="(self, query: str) -> str"
    )    
    def simple_query(
        self,
        query: str,
        vlm_model: Literal[
            "gpt4v", 
            "gemini",
            "qwen25"
        ]="gemini",
        model_init_args: Dict[str, Any]={
            "model_name": "gemini-2.5-pro",
            "generate_config": {"temperature": 0.0}
        },
        print_msgs: bool=False,
        **kwargs,
    ):
        """
        Perform a simple VLM query on the image and mask.

        Args:
            query (str): The query string.
            vlm_model (Literal): The VLM model to use. Defaults to "gemini".
                Options are "gpt4v", "gemini", "qwen25".
            model_init_args (Dict[str, Any]): The initialization arguments for the VLM model.
                Defaults to {
                    "model_name": "gemini-2.5-pro",
                    "generate_config": {"temperature": 0.0}
                }.
            print_msgs (bool): Whether to print the messages in the prompt. Defaults to False.
        kwargs: Additional arguments to pass to the VLM model.
        Returns:
            str: The response from the VLM model.
        """
        log_event(stage="image_patch", event="simple_query", msg="Simple VLM query started", meta={
            "query": query,
            "vlm_model": vlm_model,
            "model_init_args": model_init_args,
            "print_msgs": print_msgs,
        })
        return self.vlm_query(
            query=query,
            vlm_model=vlm_model,
            prompt_template="default_simple",
            model_init_args=model_init_args,
            msg_params=None,
            print_msgs=print_msgs,
            **kwargs,
        )
    
    def forward(self,
                model_name: str,
                conversation: List[Dict[str, Any]], 
                *args, 
                init_args: Dict[str, Any]={
                    "model_name": "gemini-2.5-pro",
                    "generate_config": {"temperature": 0.0}
                }, 
                **kwargs
        ):
        return forward(
            model_name, conversation, *args, init_args=init_args, **kwargs)
    
    @staticmethod
    def _examples(version: str = "v0") -> str:
        pass
        
if __name__ == "__main__":
    
    data_dir = osp.join(root, "data")
    masks_dir = osp.join(data_dir, "segmentation-masks")
    rgb_frames_dir = osp.join(data_dir, "rgb-frames")

    furniture_type = "Chair"
    furniture_name = "mammut_1"
    video_id = "FoVtnbm0hPc"
    
    mask_fn = osp.join(
        masks_dir, furniture_type, furniture_name, video_id, f"{video_id}.json"
    )
    frame_idx = 1
    with open(mask_fn) as f:
        masks = json.load(f)["manual"]
    masks = masks[str(frame_idx)]
    # masks = {k: mask_utils.decode(v) for k, v in masks.items()}
    
    rgb_fn = osp.join(
        rgb_frames_dir, furniture_type, furniture_name, video_id, f"{frame_idx}.jpg"
    )

    frame_idx_2 = 185
    rgb_fn_2 = osp.join(
        rgb_frames_dir, furniture_type, furniture_name, video_id, f"{frame_idx_2}.jpg"
    )
    
    image_patch = ImagePatch(
        input_image=rgb_fn,
        masks=masks,
        cache_dir=osp.join(root, "tmp", "tva_cache"),
    )

    print(
        image_patch.track_and_find_all_connected_pairs(
            new_input_image=np.array(Image.open(rgb_fn_2)),
            vlm_model="qwen25_vl",
            model_init_args={
                "model_name": "Qwen/Qwen2.5-VL-32B-Instruct",
                "generate_config": {"temperature": 0.0},
                "forward_pipeline": "vllm_online"
            },
            print_msgs=True
        )
    )
    
    print(
        image_patch.vlm_query(
            query="What is shown in the image?", 
            vlm_model="gemini",
            prompt_template="default",
            model_init_args={
                "model_name": "gemini-2.5-pro",
                "generate_config": {"temperature": 0.0}
            },
            print_msgs=True,
        )
    )
    
    print(
        image_patch.check_part_connectivity(
            part_id1="0",
            part_id2="1",
            vlm_model="gemini",
            model_init_args={
                "model_name": "gemini-2.5-pro",
                "generate_config": {"temperature": 0.0}
            },
            print_msgs=True
        ),
       
    )

    print(
        image_patch.get_all_connected_part_pairs(
            vlm_model="gemini",
            model_init_args={
                "model_name": "gemini-2.5-pro",
                "generate_config": {"temperature": 0.0}
            },
            print_msgs=True
        )
    )


    # print(
    #     image_patch.track_and_find_all_connected_pairs(
    #         new_input_image=np.array(Image.open(rgb_fn_2)),
    #         vlm_model="gemini",
    #         model_init_args={
    #             "model_name": "gemini-2.5-pro",
    #             "generate_config": {"temperature": 0.0}
    #         },
    #         print_msgs=True
    #     )
    # )
    
    
