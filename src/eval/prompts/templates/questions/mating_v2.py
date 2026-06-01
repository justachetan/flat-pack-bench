from typing import Dict, List, Literal, Tuple, Any

import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import os
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

from src.benchmark.generator.common import form_question_string, check_question_answer_string
from IKEAVideo.dataloader.assembly_video import decode_mask

question_type = "mating"
question_templates = dict()
question_templates["1part"] = ["Out of the given options, which part will {query_part} be directly connected to during the assembly video?"]
question_templates["2parts"] = ["Are {query_part1} and {query_part2} connected in the fully-assembled furniture?"]
question_templates["2parts_frame_old"] = ["Are {query_part1} and {query_part2} connected (physically in contact) in the shown image?"]
question_templates["2parts_frame"] = ["In this task, \"connected\" means the two parts are in direct physical contact, in the same way they will be when the furniture is fully assembled (not merely near each other or partially aligned). Based on the given image, are {query_part1} and {query_part2} connected?"]

question_templates["find_edges"] = ["Which of the parts shown in the highlighted image are directly connected in the fully-assembled furniture?"]

template_option_type = dict()
template_option_type["1part"] = "part"
template_option_type["2parts"] = "binary"
template_option_type["find_edges"] = "edge"
template_option_type["2parts_frame"] = "binary"
template_option_type["2parts_frame_old"] = "binary"


def make_1part_qstr(
    part_graph: Dict[str, List[str]],
    qstr: str,
    seg_maps_for_frame: Dict[str, Dict[str, np.ndarray]],
    frame_subasm: List[str]=None,
    select_part_strat: Literal["random", "size_h2l", "size_l2h", "subasm"]="subasm",
    num_questions: int=3,
    add_abstain: bool = False,
    add_none_at_end: bool = False,
    question_template_type: Literal["1part", "1part_img"] = "1part",
    **kwargs
):
    """
    number of total options must be 4, otherwise forming
    options becomes very complicated
    """
    if select_part_strat == "subasm" and frame_subasm is None:
        select_part_strat = "random"
    
    num_options = 4
    none_str = "None of the options are correct"
    
    frame_parts = sorted(list(seg_maps_for_frame.keys()), key=lambda x: int(x))
    frame_parts = [j for j in frame_parts if decode_mask(seg_maps_for_frame[j]).any()]
    num_questions = min(num_questions, len(frame_parts))
    
    if len(frame_parts) <= 1:
        return []
    
    if select_part_strat == "random":
        parts_for_q = [frame_parts[i] for i in \
                       np.random.choice(np.arange(len(frame_parts)), size=num_questions, replace=False)]
    elif "size" in select_part_strat:
        frame_parts_by_size = sorted(frame_parts, 
                                     key=lambda x: decode_mask(
                                         seg_maps_for_frame[x]).astype(int).sum(), 
                                     reverse=True if select_part_strat == "size_h2l" else False
                                    )
        parts_for_q = frame_parts_by_size[:num_questions]
    elif "subasm" in select_part_strat:
        # bias sampling towards parts that are part of smaller subassemblies
        subasm_idx = [subasm_idx for part_id in frame_parts for subasm_idx in \
            range(len(frame_subasm)) if part_id in frame_subasm[subasm_idx].split(",")]
        size_of_subasm = {frame_parts[i]: len(frame_subasm[subasm_idx[i]].split(",")) for i in range(len(frame_parts))}
        frame_parts_by_subasm_size = sorted(frame_parts,
                             key=lambda x: size_of_subasm[x])
        parts_for_q = frame_parts_by_subasm_size[:num_questions]
        
    
    questions = list()
    filled_qstr = copy.deepcopy(qstr)
    num_parts = len(frame_parts)
    
    for part_idx, part_id in enumerate(parts_for_q):
        
        question = dict()
        
        connected_parts = [p for p in part_graph[part_id] if p in frame_parts]
        disconnected_parts = list(set(frame_parts) - set(connected_parts + [part_id]))
        
        if len(connected_parts) + len(disconnected_parts) < num_options - 2:
            continue
        
        # num_cp = len(connected_parts)
        # num_dp = len(disconnected_parts)
        
        question_template_params = {"query_part": f"Part {part_id}"}
        
        question = form_question_string(
            question_template=filled_qstr,
            correct_answers=connected_parts,
            incorrect_answers=disconnected_parts,
            named_params_for_question=question_template_params,
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
            correct_answers=connected_parts,
            incorrect_answers=disconnected_parts,
            template_specific_check=None
        )
        
        questions.append(question)
        
    return questions

def make_2part_qstr(
    part_graph: Dict[str, List[str]],
    qstr: str,
    seg_maps_for_frame: Dict[str, Dict[str, np.ndarray]],
    frame_subasm: List[str]=None,
    select_part_strat: Literal["random", "size_h2l", "size_l2h", "subasm"]="subasm",
    num_questions: int=3,
    add_abstain: bool = False,
    add_none_at_end: bool = False,
    question_template_type: Literal["2parts", "2parts_img"] = "2parts",
    **kwargs
):
    if select_part_strat == "subasm" and frame_subasm is None:
        select_part_strat = "random"
    num_options = 2
    
    frame_parts = sorted(list(seg_maps_for_frame.keys()), key=lambda x: int(x))
    frame_parts = [j for j in frame_parts if decode_mask(seg_maps_for_frame[j]).any()]
    num_questions = min(num_questions, len(frame_parts))
    
    frame_part_pairs = [pair for pair in itertools.combinations(frame_parts, r=2)]
    part_pairs_for_q = list()
    if select_part_strat == "random":        
        
        rdm_part_idx = np.random.choice(np.arange(len(frame_part_pairs)), size=num_questions, replace=False)
        part_pairs_for_q = [frame_part_pairs[j] for j in rdm_part_idx]
        
    elif "size" in select_part_strat:
        frame_parts_by_size = sorted(frame_part_pairs, 
                                     key=lambda x: sum(
                                         [decode_mask(
                                             seg_maps_for_frame[x[xx]]).astype(int).sum() 
                                          for xx in range(len(x))
                                         ]
                                     ),
                                     reverse=True if select_part_strat == "size_h2l" else False
                                    )
        part_pairs_for_q = frame_parts_by_size[:num_questions]
    elif "subasm" in select_part_strat:
        # bias sampling towards pairs that are not already connected and belong to smaller subassemblies
        subasm_idx = {part_id: subasm_idx for part_id in frame_parts for subasm_idx in \
            range(len(frame_subasm)) if part_id in frame_subasm[subasm_idx].split(",")}
        size_of_subasm = {part_id: len(frame_subasm[subasm_idx[part_id]].split(",")) if part_id in subasm_idx else 0 for part_id in frame_parts}
        frame_parts_by_subasm_size = sorted(frame_part_pairs,
                             key=lambda x: size_of_subasm[x[0]] + size_of_subasm[x[1]])
        is_alread_connected = lambda pair: pair[0] in subasm_idx and pair[1] in subasm_idx and \
            subasm_idx[pair[0]] == subasm_idx[pair[1]]
        already_connected_pairs = [pair for pair in frame_parts_by_subasm_size if is_alread_connected(pair)]
        not_already_connected_pairs = [pair for pair in frame_parts_by_subasm_size if not is_alread_connected(pair)]
        part_pairs_for_q = not_already_connected_pairs + already_connected_pairs
        # this way we sort the pairs by the size of the subassemblies they belong to
        # and also prefer those which are NOT already connected as per the current subassembly
        # annotations
        part_pairs_for_q = frame_parts_by_subasm_size[:num_questions]
    
    questions = list()
    filled_qstr = copy.deepcopy(qstr)

    for part_idxs, part_id_pair in enumerate(part_pairs_for_q):
        
        question = dict()
        correct_answers = list()
        incorrect_answers = list()
        
        if part_id_pair[1] in part_graph[part_id_pair[0]]:
            assert part_id_pair[0] in part_graph[part_id_pair[1]], "graph edges do not appear symmetric"
            correct_answers.append("Yes")
            incorrect_answers.append("No")
        else:
            assert part_id_pair[0] not in part_graph[part_id_pair[1]], "graph edges do not appear symmetric"
            correct_answers.append("No")
            incorrect_answers.append("Yes")
        
        part_id_pair = list(part_id_pair)
        random.shuffle(part_id_pair)
        question_template_params = {
            "query_part1": f"Part {part_id_pair[0]}",
            "query_part2": f"Part {part_id_pair[1]}"
        }
        
        question = form_question_string(
            question_template=filled_qstr,
            correct_answers=correct_answers,
            incorrect_answers=incorrect_answers,
            named_params_for_question=question_template_params,
            question_type=question_type,
            question_template_type=question_template_type,
            option_type="",
            capitalize_option_type=True,
            yes_and_no_question=True,
            add_abstain=add_abstain,
            add_none_at_end=add_none_at_end,
        )
        
        is_question_correct = check_question_answer_string(
            question_dict=question,
            num_options=num_options,
            correct_answers=correct_answers,
            incorrect_answers=incorrect_answers,
            template_specific_check=None
        )
        
        questions.append(question)
        
        
    return questions

def make_find_edges_qstr(
    part_graph: Dict[str, List[str]],
    qstr: str,
    seg_maps_for_frame: Dict[str, Dict[str, np.ndarray]],
    frame_subasm: List[str]=None,
    add_abstain: bool = False,
    add_none_at_end: bool = False,
    question_template_type: Literal["find_edges", "find_edges_w_img"] = "find_edges",
    **kwargs
):
    """
    TODO: find a way to allow find_edges to use frame_subasm to include those edges in options
    that have not yet been formed (making use of the subassembly annotations in frame_subasm)
    """
    num_options = 4
    questions = list()
    
    frame_parts = sorted(list(seg_maps_for_frame.keys()), key=lambda x: int(x))
    frame_parts = [j for j in frame_parts if decode_mask(seg_maps_for_frame[j]).any()]
    
    num_parts = len(frame_parts)
    
    edges = [[k, v] for k in part_graph for v in part_graph[k] if k < v ]
    
    edges = [x for x in edges if (x[0] in frame_parts and x[1] in frame_parts)]
    
    is_edge = lambda x: x[0] in part_graph[x[1]] and x[1] in part_graph[x[0]]
    non_edges = [sorted(x, key=lambda y: int(y)) 
                 for x in itertools.combinations(frame_parts, 2) if not is_edge(x)]
    
    num_edges = len(edges)
    num_clique_edges = len(edges) + len(non_edges)
    # import ipdb; ipdb.set_trace()
    # import ipdb; ipdb.set_trace()
    if num_clique_edges < 2:
        return []

    """
    NOTE:
    in the `find_edges` template, if we want to form multiple questions,
    we need to ensure that the edges and non-edges in the options are not repeated.
    
    This is an expensive task, since we would have to store the tuple of selected
    edges and non-edges in the options for each question. To make sure things don't
    get repeated and we still get to the maximum number of question quickly, we would
    have to enumerate all such edge-set and non-edge-set pairs, and check membership.
    
    This is quite expensive, and so we don't use max_num_questions as a parameter here.
    We pass all edges and non-edges to the `form_question_string` function, and let it
    compute one question for the frame.
    """


    question = form_question_string(
        question_template=qstr,
        correct_answers=edges,
        incorrect_answers=non_edges,
        named_params_for_question=None,
        question_type=question_type,
        question_template_type=question_template_type,
        add_abstain=add_abstain,
        add_none_at_end=add_none_at_end,
    )

    is_question_correct = check_question_answer_string(
        question_dict=question,
        num_options=num_options,
        correct_answers=edges,
        incorrect_answers=non_edges,
        template_specific_check=None
    )

    return [question]


def make_mating_questions_by_type(
    part_graph: Dict[str, List[str]],
    qstr: str,
    seg_maps_for_frame: Dict[str, np.ndarray],
    template_type: Literal["1part", "2parts", "find_edges"],
    select_part_strat: Literal["random", "size_h2l", "size_l2h"]="random",
    num_questions: int=3,
    add_abstain: bool = False,
    add_none_at_end: bool = False,
    **kwargs,
):
    if template_type in ["1part", "1part_img"]:

        questions = make_1part_qstr(
            part_graph=part_graph,
            qstr=qstr,
            seg_maps_for_frame=seg_maps_for_frame,
            select_part_strat=select_part_strat,
            num_questions=num_questions,
            add_abstain=add_abstain,
            question_template_type=template_type,
            add_none_at_end=add_none_at_end,
            **kwargs
        )

    elif template_type in ["2parts", "2parts_img"]:
        questions = make_2part_qstr(
            part_graph=part_graph,
            qstr=qstr,
            seg_maps_for_frame=seg_maps_for_frame,
            select_part_strat=select_part_strat,
            num_questions=num_questions,
            add_abstain=add_abstain,
            question_template_type=template_type,
            add_none_at_end=add_none_at_end,
            **kwargs
        )
        
    elif template_type in ["find_edges", "find_edges_w_img"]:
        questions = make_find_edges_qstr(
            part_graph=part_graph,
            qstr=qstr,
            seg_maps_for_frame=seg_maps_for_frame,
            num_questions=num_questions,
            add_abstain=add_abstain,
            question_template_type=template_type,
            add_none_at_end=add_none_at_end,
            **kwargs
        )
    else:
        raise NotImplementedError(
            f"question template {template_type} not implemented")
    return questions
    

def make_mating_questions(
    part_graph: Dict[str, List[str]],
    question_templates: Dict[str, List[str]],
    seg_maps_for_frame: Dict[str, Dict[str, np.ndarray]],
    select_part_strat: Literal["random", "size_h2l", "size_l2h"]="random",
    num_questions: int=3,
    add_abstain: bool = False,
    **kwargs
):
    
    questions = dict()
    for template_type in question_templates:
        questions[template_type] = list()
        for i in range(len(question_templates[template_type])):
            
            if template_type == "1part":
                questions[template_type].extend(make_1part_qstr(
                    part_graph=part_graph,
                    qstr=question_templates[template_type][i],
                    seg_maps_for_frame=seg_maps_for_frame,
                    select_part_strat=select_part_strat,
                    num_questions=num_questions,
                    add_abstain=add_abstain
                ))
            elif template_type == "2parts":
                questions[template_type].extend(make_2part_qstr(
                    part_graph=part_graph,
                    qstr=question_templates[template_type][i],
                    seg_maps_for_frame=seg_maps_for_frame,
                    select_part_strat=select_part_strat,
                    num_questions=num_questions,
                    add_abstain=add_abstain
                ))
            elif template_type == "find_edges":
                questions[template_type].extend(
                    make_find_edges_qstr(
                        part_graph=part_graph,
                        qstr=question_templates[template_type][i],
                        seg_maps_for_frame=seg_maps_for_frame,
                        num_questions=num_questions,
                        add_abstain=add_abstain
                ))
            else:
                raise NotImplementedError(
                    f"question template {template_type} not implemented")

    return questions


