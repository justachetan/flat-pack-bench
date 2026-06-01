import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from typing import Dict, Union, Any
import os
import os.path as osp
import json
import copy
import yaml
import random
from itertools import permutations
from dataclasses import dataclass
from pathlib import Path

import omegaconf
from omegaconf import DictConfig, OmegaConf

import numpy as np
import scipy as sp
import pandas as pd

from src.eval.prompts.media.pipeline import MediaPipeline
from src.eval.prompts.templates.questions.common import calculate_sha256
from src.eval.prompts.templates.questions.convert_yaml_to_json import (
    convert_yaml_to_json,
)

@dataclass
class Template:
    """A class representing a template for prompts."""
    
    # name of the template
    name: str
    
    # description of the template
    description: str
    
    # base template file name
    base_template_fn: str
    
    # custom templates for specific question categories
    custom_template_fn: Dict[str, str] = None

@dataclass
class ConvTemplateRegistry:
    
    # TODO: figure out a better way to handle TEMPLATE_DIR
    TEMPLATE_DIR = osp.join(root, "src", "eval",
                            "prompts", "templates", "conversations")
    
    SEP_MEDIA_FIRST = Template(
        name="separate_media_first",
        description="Separate media components with video first, then visual prompt.",
        base_template_fn="separate_media_first.yaml",
        custom_template_fn={
            "tracking": "separate_media_first_tracking.yaml",
        }   
    )
    
    ONLY_IMAGE_FIRST = Template(
        name="only_image_first",
        description="Only visual prompt placed at the start of the conversation. No video.",
        base_template_fn="only_image_first.yaml",
        custom_template_fn={
            "tracking": "only_image_first_tracking.yaml",
        }
    )
    
    ONLY_IMAGE_FIRST_PROACTIVE_COMMON = Template(
        name="only_image_first_proactive_common",
        description="Only visual prompt placed at the start of the conversation, proactive, common template for all.",
        base_template_fn="only_image_first_proactive_common.yaml",
        custom_template_fn={
            "tracking": "only_image_first_proactive_common_tracking.yaml"
        }
    )
    
    SEP_MEDIA_FIRST_PROACTIVE = Template(
        name="separate_media_first_proactive",
        description="Separate media components with video first, then visual prompt, proactive.",
        base_template_fn="separate_media_first_proactive_tord.yaml",
        custom_template_fn={
            "tracking": "separate_media_first_proactive_tracking.yaml",
            "mating": "separate_media_first_proactive_mating.yaml",
            "temporal_ord": "separate_media_first_proactive_tord.yaml",
            "temporal_loc": "separate_media_first_proactive_tloc.yaml",
        }
    )
    
    SEP_MEDIA_FIRST_PROACTIVE_COMMON = Template(
        name="separate_media_first_proactive_common",
        description="Separate media components with video first, then visual prompt, proactive, common template for all.",
        base_template_fn="separate_media_first_proactive_common.yaml",
        custom_template_fn={
            "tracking": "separate_media_first_proactive_common_tracking.yaml"
        }
    )
    
    COLLAGE_VP_LEFT_MEDIA_FIRST = Template(
        name="collage_vp_left_media_first",
        description="Collage visual prompt on the left, with video on the right.",
        base_template_fn="collage_vp_left.yaml",
        custom_template_fn={
            "tracking": "collage_vp_left_tracking.yaml",
        }
    )
    
    COLLAGE_VP_LEFT_MEDIA_FIRST_PROACTIVE_COMMON = Template(
        name="collage_vp_left_proactive_common",
        description="Collage visual prompt on the left, with video on the right, proactive task instructions, common template for all tasks.",
        base_template_fn="collage_vp_left_proactive_common.yaml",
        custom_template_fn={
            "tracking": "collage_vp_left_proactive_common_tracking.yaml"
        }
    )
    
    CONCAT_VP_FIRST_MEDIA_FIRST = Template(
        name="concat_vp_first",
        description="Concatenate visual prompt to the beginning of the video. Place the video in the beginning",
        base_template_fn="concat_vp_first.yaml",
        custom_template_fn={
            "tracking": "concat_vp_first_tracking.yaml"
        }
    )
    
    CONCAT_VP_FIRST_MEDIA_FIRST_PROACTIVE_COMMON = Template(
        name="concat_vp_first_proactive_common",
        description="Concatenate visual prompt to the beginning of the video. Place the video in the beginning, proactive task instructions, common template for all tasks.",
        base_template_fn="concat_vp_first_proactive_common.yaml",
        custom_template_fn={
            "tracking": "concat_vp_first_proactive_common_tracking.yaml"
        }
    )

    CONCAT_VP_FIRST_MEDIA_FIRST_PROACTIVE_COMMON_WITH_THOUGHTS_V2 = Template(
        name="concat_vp_first_proactive_common_with_thoughts_v2",
        description="Concatenate visual prompt to the beginning of the video. Place the video in the beginning, proactive task instructions, common template for all tasks, and ask for thoughts in response.",
        base_template_fn="concat_vp_first_proactive_common_with_thoughts_v2.yaml",
        custom_template_fn={
            "tracking": "concat_vp_first_proactive_common_tracking_with_thoughts_v2.yaml"
        }
    )
    
    SEP_MEDIA_FIRST_WITH_THOUGHTS = Template(
        name="separate_media_first_with_thoughts",
        description="Separate media components with video first, then visual prompt, and ask for thoughts in response.",
        base_template_fn="separate_media_first_with_thoughts.yaml",
        custom_template_fn={
            "tracking": "separate_media_first_with_thoughts_tracking.yaml",
        }
    )
    
    SEP_MEDIA_FIRST_PROACTIVE_COMMON_WITH_THOUGHTS = Template(
        name="separate_media_first_proactive_common_with_thoughts",
        description="Separate media components with video first, then visual prompt, proactive, common template for all, and ask for thoughts in response.",
        base_template_fn="separate_media_first_proactive_common_with_thoughts.yaml",
        custom_template_fn={
            "tracking": "separate_media_first_proactive_common_tracking_with_thoughts.yaml"
        }
    )
    
    SEP_MEDIA_FIRST_PROACTIVE_COMMON_WITH_THOUGHTS_V2 = Template(
        name="separate_media_first_proactive_common_with_thoughts_v2",
        description="Separate media components with video first, then visual prompt, proactive, common template for all, and ask for thoughts in response.",
        base_template_fn="separate_media_first_proactive_common_with_thoughts_v2.yaml",
        custom_template_fn={
            "tracking": "separate_media_first_proactive_common_tracking_with_thoughts_v2.yaml"
        }
    )
    
    
    def __call__(self, key: str) -> Template:
        return getattr(self, key)
         

class TemplateBuilder:
    
    def __init__(
        self,
        video_dir: str,
        mask_dir: str,
        img_dir: str,
        media_pipeline_cfg: DictConfig,
        media_cache_dir: str,
        num_shuffles: int = 5,
        **kwargs
    ):
        """TemplateBuilder for constructing prompts with media components.

        Args:
            video_dir (str): Directory containing video files.
            mask_dir (str): Directory containing mask files.
            img_dir (str): Directory containing image files.
            media_pipeline_cfg (DictConfig): Configuration for the media pipeline.
            media_cache_dir (str): Directory for caching media files.
            num_shuffles (int, optional): Number of shuffles for the options. 0 means no shuffling. Defaults to 5.
        """
        
        # directories with the raw data
        self.video_dir = video_dir
        self.mask_dir = mask_dir
        self.img_dir = img_dir
        
        self.media_cache_dir = media_cache_dir
        self.media_pipeline_cfg = media_pipeline_cfg
        self.media_pipeline = MediaPipeline(
            cfg=media_pipeline_cfg, 
            media_cache_dir=media_cache_dir
        )
        
        self.conv_template_registry = ConvTemplateRegistry()
        self.conv_template = self.conv_template_registry(
            media_pipeline_cfg.conv_template)
        
        self.num_shuffles = num_shuffles
    
    def _load_template_file(self, template: str) -> dict[str, Any]:
        """Load a template file from the templates directory."""
        
        template_path = osp.join(
            ConvTemplateRegistry.TEMPLATE_DIR, template)
        
        if not osp.exists(template_path):
            raise FileNotFoundError(f"Template file {template_path} does not exist.")
        
        with open(template_path, "r") as f:
            # import ipdb; ipdb.set_trace()
            return OmegaConf.to_container(OmegaConf.load(f),
                resolve=True, throw_on_missing=False)

    def _render_conversation(
        self, 
        question_json: dict[str, Any],
        out_name: Union[str, Path],
        rendered_template_dir: str,
        override_cache: bool = False,
        jumble_map: Dict[str, str] = None,
        **kwargs
    ):
        
        # TODO: run the media pipeline
        category = question_json["vid_category"]
        name = question_json["furniture_name"]
        vid = question_json["video_id"]
        frame_idx = question_json["frame_idx"]
        # jumble_map = question_json.get("jumble_map", None)
        
        video_path = osp.join(self.video_dir, category, name, vid, f"{vid}.mp4")
        mask_path = osp.join(self.mask_dir, category, name, vid, f"{vid}.json")
        img_dir = osp.join(self.img_dir, category, name, vid)
        
        # check if approproate media files exist
        rendered_media = self.media_pipeline.run(
            video_path=video_path,
            mask_path=mask_path,
            img_dir=img_dir,
            frame_idxs=frame_idx,
            jumble_map=jumble_map,
            colors=None,
            edge_colors=None,
            override_cache=override_cache,
        )
        # import ipdb; ipdb.set_trace()
                
        # create the conversation template, take care to 
        # store the options separate from the question step so that
        # shuffling of the options is possible
        question_category = question_json["question_category"]
        # import ipdb; ipdb.set_trace()
        conv_template = None
        if question_category in self.conv_template.custom_template_fn:
            conv_template = self._load_template_file(
                self.conv_template.custom_template_fn[question_category])
        else:
            conv_template = self._load_template_file(
                self.conv_template.base_template_fn)
        
        media_cache_dir = self.media_cache_dir
        question_json["media_dir"] = media_cache_dir
        for msg in conv_template:
            # TODO: incorporate video clips
            if msg["type"] == "video":
                
                msg["content"] = osp.join(
                    media_cache_dir,
                    rendered_media["video"]
                )
                question_json["video"] = rendered_media["video"]
                
            elif msg["tag"] == "visual_prompt":
                visual_prompt_fn = rendered_media["visual_prompts"][rendered_media["vp_frame_idxs"][0]]
                msg["content"] = osp.join(
                    media_cache_dir,
                    visual_prompt_fn
                )
                question_json_visual_prompt_key = "prompt_img_fn"
                if question_json_visual_prompt_key not in question_json:
                    question_json_visual_prompt_key = "prompt_img_0_fn"
                question_json[question_json_visual_prompt_key] = visual_prompt_fn
                
            elif msg["tag"] == "jumbled_visual_prompt":
                visual_prompt_fn = rendered_media["visual_prompts"][rendered_media["vp_frame_idxs"][-1]]
                msg["content"] = osp.join(
                    media_cache_dir,
                    visual_prompt_fn
                )
                question_json["jumbled_prompt_img_1_fn"] = visual_prompt_fn
            elif msg["tag"] == "question":
                msg["content"] = question_json["question"]["qstr"]
        
        # dump metadata and skip dumping of conversation if it already exists
        # TODO: optimize this clash checking to have it early on in the template rendering
        question_matadata = copy.deepcopy(question_json)
        question_matadata.pop("question")
        
        question_metadata_db = dict()
        if osp.exists(osp.join(rendered_template_dir, "question_metadata.json")):
            with open(osp.join(rendered_template_dir, "question_metadata.json"), "r") as f:
                question_metadata_db = json.load(f)
        question_in_metadata_db = question_matadata["qid"] in question_metadata_db 
        if question_in_metadata_db:
            old_qid = question_matadata["qid"]
        for k, v in question_metadata_db.items():
            is_match = (question_matadata["qid_flat"] == v["qid_flat"]) or \
                question_matadata["qid_flat"].split("/")[-2] == v["qid_flat"].split("/")[-2]
            if is_match:
                question_in_metadata_db = True
                old_qid = v["qid"]
                assert old_qid == k, f"Metadata DB has inconsistent qid for {question_matadata['qid_flat']}"
                break
        if (not question_in_metadata_db) or override_cache:
            if question_in_metadata_db:
                if old_qid != question_matadata["qid"]:
                    question_metadata_db.pop(old_qid, None)
            question_metadata_db[question_matadata["qid"]] = question_matadata
            with open(osp.join(rendered_template_dir, "question_metadata.json"), "w") as f:
                json.dump(question_metadata_db, f, indent=4)
        
        # NOTE: will facilitate writing a viewer for the generated questions
        all_rendered_questions = []
        if osp.exists(osp.join(rendered_template_dir, "questions.jsonl")):
            # NOTE: not the best idea but number of questions is small enough to do this
            with open(osp.join(rendered_template_dir, "questions.jsonl"), "r") as f:
                all_rendered_questions = [json.loads(line.strip()) for line in f if line.strip()]
        in_cache_idx = -1
        for loaded_question_idx in range(len(all_rendered_questions)):
            is_match = all_rendered_questions[loaded_question_idx]["qid"] == question_json["qid"] or \
                all_rendered_questions[loaded_question_idx]["qid_flat"].split("/")[-2] == question_json["qid_flat"].split("/")[-2]
            if is_match:
                # all_rendered_questions[loaded_question_idx] = question_json
                in_cache_idx = loaded_question_idx
                break
        if in_cache_idx >= 0 and override_cache:
            all_rendered_questions[in_cache_idx] = question_json
        elif in_cache_idx < 0:
            # if the question is not in the cache, append it
            all_rendered_questions.append(question_json)

        with open(osp.join(rendered_template_dir, "questions.jsonl"), "w") as f:
            for q in all_rendered_questions:
                f.write(json.dumps(q) + "\n")


        conversation_metadata = {}
        if osp.exists(osp.join(rendered_template_dir, "conversation_metadata.json")):
            with open(osp.join(rendered_template_dir, "conversation_metadata.json"), "r") as f:
                conversation_metadata = json.load(f)
            # import ipdb; ipdb.set_trace()
            if out_name in conversation_metadata and (not override_cache):
                return {
                    "conversation_metadata": {
                        out_name: {
                            "qid": question_json["qid"],
                            "qid_flat": question_json["qid_flat"],
                            "clash": out_name in conversation_metadata,
                        }
                    },
                    "question_metadata": {
                        question_matadata["qid"]: question_matadata,
                    }
                }
        
        if override_cache:
            for conv_id in list(conversation_metadata.keys()):
                cache_qid_flat = conversation_metadata[conv_id]["qid_flat"]
                question_yaml_for_cache_conv = cache_qid_flat.split("/")[-2]
                if question_json["qid_flat"].split("/")[-2] == question_yaml_for_cache_conv:
                    # remove the conversation metadata for the old question
                    conversation_metadata.pop(conv_id, None)        
            
        conversation_metadata[out_name] = {
            "qid": question_json["qid"],
            "qid_flat": question_json["qid_flat"],
            "clash": out_name in conversation_metadata,
        }

        with open(osp.join(rendered_template_dir, "conversation_metadata.json"), "w") as f:
            json.dump(conversation_metadata, f, indent=4)
        
        
        os.makedirs(osp.join(rendered_template_dir, out_name), exist_ok=True)
        with open(osp.join(rendered_template_dir, out_name, "conversation.yaml"), "w") as f:
            yaml.dump(conv_template, f)
        with open(osp.join(rendered_template_dir, out_name, "question.json"), "w") as f:
            json.dump(question_json, f, indent=4)
        
        return {
            "conversation_metadata": {
                out_name: {
                    "qid": question_json["qid"],
                    "qid_flat": question_json["qid_flat"],
                    "clash": out_name in conversation_metadata,
                }
            },
            "question_metadata": {
                question_matadata["qid"]: question_matadata,
            }
        }
    
    def build(
        self, 
        rendered_template_dir: str, 
        yaml_fn: str, 
        override_cache: bool = False,
        question_jsonl_fn: str = None,
        **kwargs
    ):
        """
        Build the prompt using the specified question YAML.
        
        Args:
            rendered_template_dir (str): Directory to store the rendered templates.
            yaml_fn (str): Path to the question YAML file.
            question_jsonl_fn (str): Path to the question JSONL file. Defaults to None.
        """
        
        os.makedirs(rendered_template_dir, exist_ok=True)
        
        # Load the question YAML
        with open(yaml_fn, "r") as f:
            base_question_yaml = yaml.safe_load(f)
        
        num_shuffles = 1
        # if shuffling is enabled,
        if self.num_shuffles >= 1:
            # NOTE: this conditional block is redundant as by default we assume 1 shuffle. Just keeping it here for legacy.
            # TODO: refactor this to remove the redundancy.
            num_shuffles = self.num_shuffles
            option_idxs = list(range(len(base_question_yaml["options"])))
            
            # TODO: check if we should clip number of shuffles
            #       by (#options)! (factorial of number of options)
            # import ipdb; ipdb.set_trace()
            # TODO: error here. need to fix
            # found some cases where correct option was not correct
            # and query parameters are getting incorrectly set
            
            for shuff_iter_idx, shuffled_idxs in enumerate(permutations(option_idxs, len(option_idxs))):
                
                if shuff_iter_idx >= num_shuffles:
                    break
                question_yaml = copy.deepcopy(base_question_yaml)
                # shuffled_idxs = copy.deepcopy(option_idxs)
                # import ipdb; ipdb.set_trace()
                # shuffled_idxs = np.random.permutation(shuffled_idxs).tolist()
                options = copy.deepcopy(question_yaml["options"])
                shuffled_options = [options[i] for i in shuffled_idxs]
                question_yaml["options"] = shuffled_options
                if isinstance(question_yaml["correct_option"]["idx"], int):
                    question_yaml["correct_option"]["idx"] = shuffled_idxs.index(question_yaml["correct_option"]["idx"])
                elif isinstance(question_yaml["correct_option"]["idx"], list):
                    # shuffle the indices in the correct option
                    question_yaml["correct_option"]["idx"] = [
                        shuffled_idxs.index(i) for i in question_yaml["correct_option"]["idx"]
                    ]
                
                # first convert YAML to JSON
                question_json = convert_yaml_to_json(question_yaml)
                # add the qid with the shuff_iter_idx to the question_json
                # NOTE: this is important as it will prevent re-rendering of the same question
                question_json["qid_flat"] = question_json["qid_flat"] + "/" + osp.basename(yaml_fn).split('.')[0] + \
                    "/" + str(shuff_iter_idx)
                question_json["qid"] = calculate_sha256(question_json["qid_flat"])
                if question_jsonl_fn is not None:
                    
                    cached_que_json = None
                    with open(question_jsonl_fn, "r") as f:
                        for line in f:
                            line_que_json = json.loads(line.strip())
                            # NOTE (ac): did this because in some cases overrides also need to be applied when
                            #    the template, and thus qid_flat for the same question YAML has changed.
                            is_match = (question_json["qid_flat"] == line_que_json["qid_flat"]) or \
                                osp.basename(yaml_fn).split('.')[0] == line_que_json["qid_flat"].split("/")[-2]
                            if is_match:
                                cached_que_json = line_que_json
                                break
                    # NOTE (ac): removing the second check because if template changes, then qid_flat
                    #       and qid will also change significantly, and if we are overwriting, then in that case
                    #       we want to consider it a match so we can override it. Earlier, we wanted to ensure that
                    #       that the same question_json is being used for all runs which should still happen
                    #       for the question that are not supposed to be overwritten.
                    #       TODO: revert to this more rigorous check after the audit experiments are done
                    # assert cached_que_json is None or cached_que_json["qid"] == question_json["qid"], \
                    #     f"Cached question JSON does not match for {question_json['qid_flat']}"
                    assert cached_que_json is None or cached_que_json["qid_flat"].split("/")[-2] == question_json["qid_flat"].split("/")[-2], \
                        f"Cached question JSON does not match for {question_json['qid_flat']}"
                    # import ipdb; ipdb.set_trace()
                    question_json = cached_que_json
                
                jumble_map = question_yaml.get("jumble_map", None)
                if isinstance(jumble_map, list):
                    jumble_map_flat = {k: v for i in jumble_map for k, v in i.items()}
                    jumble_map = jumble_map_flat
                # import ipdb; ipdb.set_trace()
                media_pipeline_cfg_hash = calculate_sha256(
                    json.dumps(OmegaConf.to_container(self.media_pipeline_cfg, 
                                                      resolve=True)["pipeline"], sort_keys=True))
                # import ipdb; ipdb.set_trace()    
                out_name = f"{question_json['qid']}_{self.media_pipeline_cfg.conv_template}_{media_pipeline_cfg_hash}"
                self._render_conversation(
                    question_json,
                    out_name=out_name,
                    rendered_template_dir=rendered_template_dir,
                    override_cache=override_cache,
                    jumble_map=jumble_map,
                )
        
        """
        maybe we can store stuff as 
        category
        |-> name
            |-> video_id
                |-> media_cache
                    |-> <video-name>
                    |-> <visual-prompt-name>
                |-> questions
                    |-> <question-id>
                        |-> question_metadata.json
                        |-> rendered_template.yaml
                        
        OR (cleaner)
        |-> media_cache
            |-> <video-name>
            |-> <visual-prompt-name>
        |-> questions
            |-> <question-id>
                |-> question_metadata.json
                |-> rendered_template.yaml
        |-> question_id_metadata.csv
        |-> conversation_metadata.csv
        
        can also consider storing the question metadata in the rendered template itself
        so that we can have a single file per question.
        the rendered template can have two high-level keys, "metadata" and "conversation".
        The "metadata" key can contain the question metadata, and the "conversation" key can
        contain the conversation template.
        """
