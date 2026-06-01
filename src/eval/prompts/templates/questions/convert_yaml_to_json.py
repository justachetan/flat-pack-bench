import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from typing import Dict, List, Literal, Tuple, Any, Union

import os
import os.path as osp
import json
import glob
import copy
import yaml
import shutil

import numpy as np
import mediapy as media
import imageio.v2 as iio
import pandas as pd

from dataclasses import dataclass

from src.eval.prompts.templates.questions.mating_v2 import question_templates as mating_templates
from src.eval.prompts.templates.questions.temporal_loc import question_templates as temporal_loc_templates
from src.eval.prompts.templates.questions.temporal_ord import question_templates as temporal_ord_templates
from src.eval.prompts.templates.questions.tracking import question_templates as tracking_templates

from src.eval.prompts.templates.questions.mating_v2 import template_option_type as mating_option_type
from src.eval.prompts.templates.questions.temporal_loc import template_option_type as temporal_loc_option_type
from src.eval.prompts.templates.questions.temporal_ord import template_option_type as temporal_ord_option_type
from src.eval.prompts.templates.questions.tracking import template_option_type as tracking_option_type

from src.eval.prompts.templates.questions.common import (
    form_question_string, 
    answer_option_templates, 
    calculate_sha256
)

@dataclass
class QuestionTemplateRegistry:
    """A registry for question templates."""
    
    mating = mating_templates
    temporal_loc = temporal_loc_templates
    temporal_ord = temporal_ord_templates
    tracking = tracking_templates

    def __getitem__(self, key: str):
        """
        Allow registry lookup via indexing: registry[key] returns the templates for that key.
        """
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"No question template for category '{key}'")

    @classmethod
    def __class_getitem__(cls, key: str):
        """
        Allow class-level lookup via subscription: QuestionTemplateRegistry['mating']
        """
        try:
            return getattr(cls, key)
        except AttributeError:
            raise KeyError(f"No question template for category '{key}'")
        
@dataclass
class TemplateOptionTypeRegistry:
    """A registry for question templates."""

    mating = mating_option_type
    temporal_loc = temporal_loc_option_type
    temporal_ord = temporal_ord_option_type
    tracking = tracking_option_type

    def __getitem__(self, key: str):
        """
        Allow registry lookup via indexing: registry[key] returns the templates for that key.
        """
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(f"No question template for category '{key}'")

    @classmethod
    def __class_getitem__(cls, key: str):
        """
        Allow class-level lookup via subscription: QuestionTemplateRegistry['mating']
        """
        try:
            return getattr(cls, key)
        except AttributeError:
            raise KeyError(f"No question template for category '{key}'")

QUESTION_TEMPLATE_REGISTRY = QuestionTemplateRegistry()
TEMPLATE_OPTION_TYPE_REGISTRY = TemplateOptionTypeRegistry()

def convert_yaml_to_json(
    yaml_fn: Union[str, dict],
    **kwargs,
): 
    """
    Convert a YAML question annotation to a more complete 
    JSON annotation with the question template filled in.
    
    Args:
        yaml_fn (Union[str, dict]): Path to the YAML file or a dictionary containing question info.
    Returns:
        Dict[str, Any]: A dictionary containing the complete question information.
    """
    
    if isinstance(yaml_fn, str) and not osp.exists(yaml_fn):
        raise FileNotFoundError(f"YAML file {yaml_fn} does not exist.")

    if isinstance(yaml_fn, str):
        with open(yaml_fn, 'r') as yaml_file:
            question_info = yaml.safe_load(yaml_file)
    else:
        question_info = yaml_fn

    # for older annotations when `temporal_ord` was not used 
    if question_info["template_type"] in ["edge_order", "many_part_order"]:
        question_info["question_category"] = "temporal_ord"
    
    question_params = dict()
    if "question_params" in question_info:
        question_params = question_info["question_params"]
        for key in question_params:
            if isinstance(question_params[key], str):
                question_params[key] = [question_params[key]]
            if len(question_params[key]) == 1:
                question_params[key] = f"Part {question_params[key][0]}"
            else:
                raise RuntimeError(f"something unexpected happened in {yaml_fn}. \
                    we only have single length question params")
    
    if not isinstance(question_info["correct_option"]["idx"], list):
        question_info["correct_option"]["idx"] = [question_info["correct_option"]["idx"]]
    
    correct_options = question_info["correct_option"]["raw"]
    if len(question_info["correct_option"]["idx"]) == 1:
        correct_options = [correct_options]
    correct_options = [opt for opt in correct_options if opt != "none"]
    
    incorrect_options = [question_info["options"][i]["raw"] \
         for i in range(len(question_info["options"])) \
         if not (i in question_info["correct_option"]["idx"])]
    incorrect_options = [opt for opt in incorrect_options if opt != "none"]

    question_type = question_info["question_category"]
    template_type = question_info["template_type"]
    
    question_template = QUESTION_TEMPLATE_REGISTRY[question_type][template_type]
    if isinstance(question_template, list):
        question_template = question_template[0]
    
    num_options = 4

    if question_type in ["temporal_loc", "tracking"]:
        num_options = len(question_info["options"])
        if isinstance(question_info["correct_option"]["idx"], list) and \
            len(question_info["correct_option"]["idx"]) > 1:
            num_options += 1
        if num_options < 4 and question_type not in ["tracking"]:
            num_options += 1 # for none
            
    single_answer_type = str
    if question_type == "mating":
        single_answer_type = str
    elif question_type in ["temporal_loc", "temporal_ord"]:
        if template_type not in ["1part_order", "1part_order_last"]:
            single_answer_type = list
        if template_type in ["edge_order"]:
            single_answer_type = "listoflist"
    elif question_type == "tracking":
        single_answer_type = "listoflist"
        if template_type in ["track_single"]:
            single_answer_type = list
    
    is_binary = TEMPLATE_OPTION_TYPE_REGISTRY[question_type][template_type] == "binary"
    # import ipdb; ipdb.set_trace()
    question_dict = form_question_string(
        question_template=question_template,
        correct_answers=correct_options,
        incorrect_answers=incorrect_options,
        named_params_for_question=question_params,
        single_answer_type=single_answer_type,
        question_type=question_type,
        question_template_type=template_type,
        num_options=num_options,
        yes_and_no_question=is_binary,
        randomize_options=False,
        **kwargs,
    )
    
    qinfo = dict()
    qinfo["template_type"] = template_type
    qinfo["question_category"] = question_type
    
    qinfo["vid_category"] = question_info["category"]
    qinfo["furniture_name"] = question_info["name"]
    qinfo["video_id"] = question_info["video_id"]
    qinfo["frame_idx"] = question_info["frame_idx"]
    # TODO: update this if we start recording template_idx
    template_idx = 0
    qinfo["template_idx"] = template_idx
    
    # if a dict is directly provided, we cannot add the filename to the qid_flat
    question_str_id = "" if isinstance(yaml_fn, dict) else "/" + osp.basename(yaml_fn).split('.')[0]
    qid_flat = f"{question_type}/{template_type}/{template_idx}/{qinfo['vid_category']}/{qinfo['video_id']}{question_str_id}"

    qinfo["qid_flat"] = qid_flat
    qinfo["question"] = question_dict
    
    qid = calculate_sha256(qid_flat)
    
    if isinstance(question_info["frame_idx"], list):
        if len(question_info["frame_idx"]) == 2:
            qinfo["prompt_img_0_fn"] = f"prompt_{qinfo['frame_idx'][0]:03d}{'_q'+question_str_id[1:] if len(question_str_id)>0 else ''}.jpg"

            qinfo["jumbled_prompt_img_1_fn"] = f"jumbled_prompt_{qinfo['frame_idx'][1]:03d}{'_q'+question_str_id[1:] if len(question_str_id)>0 else ''}.jpg"
        elif len(question_info["frame_idx"]) > 2:
            raise ValueError(f"More than 2 frame indices provided for {yaml_fn}, which is not supported.")
    else:
        qinfo["prompt_img_fn"] = f"prompt_{qinfo['frame_idx']:03d}{'_q'+question_str_id[1:] if len(question_str_id)>0 else ''}.jpg"

    qinfo["qid"] = qid
    
    return qinfo
    
    
