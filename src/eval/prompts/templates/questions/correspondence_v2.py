from typing import Dict, List, Literal, Tuple, Any

import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import os
import re
import sys
import os.path as osp
sys.path.append(osp.join(root, "src"))
import copy
import json
import math
import random
import string
import itertools

import numpy as np

from IKEAVideo.dataloader.assembly_video import decode_mask
from src.benchmark.generator.common import form_question_string, check_question_answer_string

question_type = "correspondence"
question_templates = dict()
question_templates["sgc_1part"] = [
    "Out of the given options, which part will form the {semantic_role} of the fully-assembled furniture?"
]
question_templates["sgc_1part_w_cat"] = [
    "Out of the given options, which part will form the {semantic_role} of the fully-assembled {category}?"
]

# NOTE: not including "top", "bottom" as most of the furniture I annotated
# did not have top-down symmetry and I think it is pretty common to define
# a canonical top-down direction for these furniture pieces
positional_qualifiers = ["left", "right", "front", "back", "rear"]
# positiona_qualifier_regex = re.compile("|".join(map(re.escape, positional_qualifiers)))
positional_qualifier_regex = re.compile(
    r"\b(?:" + "|".join(map(re.escape, positional_qualifiers)) + r")\b\s*"
)

def erase_positional_qualifiers(semantic_role: str) -> str:
    """erase positional qualifiers from the semantic role string

    Args:
        semantic_role (str): semantic role string

    Returns:
        str: cleaned semantic role string
    """
    cleaned_semantic_role = positional_qualifier_regex.sub("", semantic_role)
    cleaned_semantic_role = re.sub(r"\s+", " ", cleaned_semantic_role)
    cleaned_semantic_role = cleaned_semantic_role.strip()
    return cleaned_semantic_role

# TODO: can we think of templates for multiple parts? such as "which of the following parts form the legs of the chair?"
#   - converting singular to plural
#   - mapping positional qualifiers to parts - map "left leg" to "legs"

def make_semantic_geometric_match_qstr(
    part_graph: Dict[str, List[str]],
    semantic_roles: Dict[str, str],
    qstr: str,
    seg_maps_for_frame: Dict[str, Dict[str, np.ndarray]],
    frame_subasm: List[str] = None,
    num_questions: int=3,
    category: str=None,
    add_abstain: bool = False,
    remove_positional_qualifiers: bool = True,
    question_template_type: Literal["sgc_1part", "sgc_1part_w_cat"] = "sgc_1part",
    add_none_at_end: bool = False,
    **kwargs
):
    
    num_options = 4
    questions = list()
    none_str = "None of the options are correct"
    
    frame_parts = sorted(list(seg_maps_for_frame.keys()), key=lambda x: int(x))
    frame_parts = [j for j in frame_parts if decode_mask(seg_maps_for_frame[j]).any()]
    # shuffle frame parts
    random.shuffle(frame_parts)
    
    clean_semantic_roles = dict() # {clean_semantic_role: frame_part}
    uniq_semantic_roles = set()
    if remove_positional_qualifiers:
        for frame_part in semantic_roles:
            semantic_role_for_frame_part = [sr.strip().lower() for sr in semantic_roles[frame_part].split(",")]
            for semantic_role in semantic_role_for_frame_part:
                cleaned_semantic_role = erase_positional_qualifiers(semantic_role)
                if cleaned_semantic_role not in clean_semantic_roles:
                    clean_semantic_roles[cleaned_semantic_role] = []
                if frame_part not in clean_semantic_roles[cleaned_semantic_role]:
                    clean_semantic_roles[cleaned_semantic_role].append(frame_part)
                uniq_semantic_roles.add(cleaned_semantic_role)
            
    uniq_semantic_roles = list(uniq_semantic_roles)
    num_semantic_roles = len(uniq_semantic_roles)
    num_questions = min(num_semantic_roles, num_questions)
    
    num_parts = len(frame_parts)
    
    if num_parts <= 2:
        return []
    
    subasm_idx_for_frame_part = None
    if frame_subasm is not None:
        # order the uniq semantic roles by ascending order of weighted
        # average of subassembly lengths of their parts
        # lesser is better 
        subasm_idx_for_frame_part = {
            frame_parts[i]: j for i in range(len(frame_parts)) for j in range(len(frame_subasm)) if frame_parts[i] in frame_subasm[j].split(",")}
        wt_avg_subasm_len_for_semantic_role = dict()
        for semantic_role in uniq_semantic_roles:
            if semantic_role not in clean_semantic_roles:
                continue
            subasm_lens = []
            for frame_part in clean_semantic_roles[semantic_role]:
                if frame_part not in subasm_idx_for_frame_part:
                    continue
                subasm_idx = subasm_idx_for_frame_part[frame_part]
                subasm_lens.append(len(frame_subasm[subasm_idx].split(",")))
            if len(subasm_lens) > 0:
                wt_avg_subasm_len_for_semantic_role[semantic_role] = sum(subasm_lens) / len(subasm_lens)
        # sort the semantic roles by their weighted average subassembly lengths
        uniq_semantic_roles = sorted(uniq_semantic_roles, key=lambda x: wt_avg_subasm_len_for_semantic_role[x] if x in wt_avg_subasm_len_for_semantic_role else 0)              
    
    for qidx in range(num_questions):            
        
        full_qstr = copy.deepcopy(qstr)
            
        # semantic_roles_for_frame_part = [sr.strip().lower() for sr in semantic_roles[frame_parts[qidx]].split(",")]
        # semantic_role = semantic_roles_for_frame_part[random.randint(0, len(semantic_roles_for_frame_part)-1)]
        
        # clean_semantic_role = semantic_role
        # if remove_positional_qualifiers:
        #     for pos_qual in positional_qualifiers:
        #         if pos_qual in semantic_role:
        #             clean_semantic_role = semantic_role.replace(pos_qual, "")

        clean_semantic_role = uniq_semantic_roles[qidx]
        all_correct_parts = [i for i in clean_semantic_roles[clean_semantic_role] if i in frame_parts]
        if frame_subasm is not None:
            # for the correct answer, only keep the parts in the frame that are part
            # of the smallest subassembly in the frame
            # NOTE: this may lead to very few question getting formed, in which case we can disable this
            # try:
            #     subasm_lens = [len(frame_subasm[subasm_idx_for_frame_part[i]].split(",")) for i in all_correct_parts if i in subasm_idx_for_frame_part]
            #     min_subasm_len = min(subasm_lens) if len(subasm_lens) > 0 else None
            # except Exception as e:
            #     import ipdb
            #     ipdb.set_trace()
            subasm_lens = [len(frame_subasm[subasm_idx_for_frame_part[i]].split(",")) for i in all_correct_parts if i in subasm_idx_for_frame_part]
            min_subasm_len = min(subasm_lens) if len(subasm_lens) > 0 else None
            
            if min_subasm_len is not None:
                all_correct_parts = [i for i in all_correct_parts \
                    if i in subasm_idx_for_frame_part and \
                        len(frame_subasm[subasm_idx_for_frame_part[i]].split(",")) == min_subasm_len]
        
        incorrect_parts = list(set(frame_parts) - set(all_correct_parts))
        
        # NOTE: if we have less than 2 possible single-answer options, skip
        if len(all_correct_parts) + len(incorrect_parts) < num_options - 2:
            continue
        
        qtemplate_params = {
            "semantic_role": clean_semantic_role,
        }
        if category is not None:
            qtemplate_params["category"] = category
            
        question = form_question_string(
            question_template=full_qstr,
            correct_answers=all_correct_parts,
            incorrect_answers=incorrect_parts,
            named_params_for_question=qtemplate_params,
            question_type=question_type,
            question_template_type=question_template_type,
            option_type="part",
            capitalize_option_type=True,
            yes_and_no_question=False,
            add_abstain=add_abstain,
            add_none_at_end=add_none_at_end,            
        )
        
        is_question_correct = check_question_answer_string(
            question_dict=question,
            num_options=num_options,
            correct_answers=all_correct_parts,
            incorrect_answers=incorrect_parts,
            template_specific_check=None
        )
        
        
        questions.append(question)
        
    return questions


def make_semantic_geometric_match_questions(
    part_graph: Dict[str, List[str]],
    semantic_roles: Dict[str, str],
    qstr: str,
    seg_maps_for_frame: Dict[str, Dict[str, np.ndarray]],
    num_questions: int=3,
    category: str=None,
    add_abstain: bool = False,
    add_none_at_end: bool = False,
    remove_positional_qualifiers: bool = True,
    **kwargs
):
    
    """
    Generate a list of questions based on the semantic roles of parts in a frame.

    Args:
        part_graph (Dict[str, List[str]]): A dictionary representing the part graph.
        semantic_roles (Dict[str, str]): A dictionary mapping part IDs to their semantic roles.
        qstr (str): The question string template.
        seg_maps_for_frame (Dict[str, Dict[str, np.ndarray]]): A dictionary containing segmentation maps for each part in the frame.
        num_questions (int): The number of questions to generate.
        category (str): The category of the furniture.
        add_abstain (bool): Whether to add an abstain option.

    Returns:
        List[Dict[str, Any]]: A list of generated questions.
    """
    # questions = list()
    
    questions = make_semantic_geometric_match_qstr(
        part_graph,
        semantic_roles,
        qstr,
        seg_maps_for_frame,
        num_questions=num_questions,
        category=category,
        add_abstain=add_abstain,
        remove_positional_qualifiers=remove_positional_qualifiers,
        add_none_at_end=add_none_at_end,
        question_template_type="sgc_1part" if category is None else "sgc_1part_w_cat",
        **kwargs
    )
            
    return questions
