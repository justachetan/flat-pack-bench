from typing import Literal, List, Dict, Any, Optional, Callable, Union
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
import copy
sys.path.append(osp.join(root, "src"))

import random
import string
import hashlib

import numpy as np

from IKEAVideo.dataloader.assembly_video import decode_mask

ABSTAIN_RAW = "abstain"
NONE_RAW = "none"

def get_cand_query_frames(preproc_seg_masks,
                          num_obj_thresh: int=2,
                          obj_area_thresh: float=10,
                          use_seg_mode: Literal["subasm", "refined", "pred"]="refined"):
    
    seg_mode = None
    if use_seg_mode == "refined":
        seg_mode = "refined_part_mask"
    elif use_seg_mode == "pred":
        seg_mode = "prediction"
    seg_masks = preproc_seg_masks[seg_mode]
    visible_pts_in_frame = {k: ([v for v in seg_masks[k] 
                                 if decode_mask(
                                     seg_masks[k][v]).astype(float).sum() > obj_area_thresh]) 
                            for k in seg_masks}
    cand_frame_indices = [
        k for k in visible_pts_in_frame if len(visible_pts_in_frame[k]) >= num_obj_thresh]
    return cand_frame_indices, visible_pts_in_frame

def calculate_sha256(input_string):        
    return hashlib.sha256(input_string.encode()).hexdigest()


class answer_option_templates:
    """Class to hold templates for answer choices in question generation
    
    NOTE: only gives the raw option string. Things like separator between options and
    the option label (e.g., "A", "B", etc.) are handled in the form_question_string function.
    
    NOTE: for cases where each single answer is not a string, referring to multiple single
            answers in the option string will become verbose. To avoid this, we are giving option
            type as "option" instead of "part" in the form_question_string function, so we will
            handle only that case here
    """

    @classmethod
    def none_str(self, mode: int = 1):
        """create option string for no answer choices

        Args:
            mode (int, optional): indicated version of the no answer string to use. Defaults to 1.
        """
        if mode == 1:
            return "None of the options are correct"
        elif mode == 2:
            return "None of the above"
        else:
            raise NotImplementedError(f"mode {mode} not implemented")
    
    @classmethod
    def abstain_str(self, mode: int = 1):
        """create option string for abstain answer choices

        Args:
            mode (int, optional): indicated version of the abstain answer string to use. Defaults to 1.
        """
        if mode == 1:
            return "Not sure"
        if mode == 2:
            return "Can't tell"
        elif mode == 2:
            return "I don't know"
        else:
            raise NotImplementedError(f"mode {mode} not implemented")
    
    @classmethod
    def more_than_two_answers(self, answers: List[str], option_type: str = "part", capitalize_option_type: bool = True, mode: int = 1):
        """create option string for more than one answer choices

        Args:
            answers (List[str]): list of answers to build the option string
            option_type (str, optional): tells how to refer to the option. Defaults to "part".
            capitalize_option_type (bool, optional): indicates if the option type should be capitalized. Defaults to True.
            mode (int, optional): indicated version of the >2 answer string to use. Defaults to 1.
        """
        if not isinstance(option_type, list):
            option_type = [option_type] * len(answers)
        if mode == 1:
            cap_opt = [opt.capitalize() for opt in option_type if capitalize_option_type]
            return f"All of {', '.join([f'{cap_opt[i]} {answers[i]}' for i in range(0, len(answers)-1)])} and {cap_opt[-1]} {answers[-1]} and are correct"
        elif mode == 2:
            cap_opt = [opt.capitalize() for opt in option_type if capitalize_option_type]
            if len(set(cap_opt)) != 1:
                raise ValueError(f"mode {mode} expects option_type {option_type} to be same for all answers")
            # cap_opt = option_type.capitalize() if capitalize_option_type else option_type
            return f"All of {cap_opt}s {', '.join(answers[:-1])} and {answers[-1]} are correct"
        else:
            raise NotImplementedError(f"mode {mode} not implemented")

    @classmethod
    def two_answers(
        self, answers: List[str], option_type: str = "part", capitalize_option_type: bool = True, mode: int = 1):
        """create option string for two answer choices

        
        Args:
            answers (List[str]): list of answers to build the option string
            option_type (str, optional): tells how to refer to the option. Defaults to "part".
            capitalize_option_type (bool, optional): indicates if the option type should be capitalized. Defaults to True.
            mode (int, optional): indicated version of the 2 answer string to use. Defaults to 1.
        """
        # cap_opt = option_type.capitalize() if capitalize_option_type else option_type
        if not isinstance(option_type, list):
            option_type = [option_type] * len(answers)
        if mode == 1:    
            cap_opt = [opt.capitalize() for opt in option_type if capitalize_option_type]
            return f"Both {cap_opt[0]} {answers[0]} and {cap_opt[1]} {answers[1]} are correct"
        elif mode == 2:
            if option_type[0] != option_type[1]:
                raise ValueError(f"mode {mode} expects option_type {option_type} to be same for both answers")
            return f"Both of {option_type}s {answers[0]} and {answers[1]} are correct"
        else:
            raise NotImplementedError(f"mode {mode} not implemented")

    @classmethod
    def single_answer(
        self, 
        answers: List[Any], 
        option_type: Union[str, List[str]] = "part",
        capitalize_option_type: bool = True, 
        mode: int = 1
    ):
        """create option string for single answer choice

        Args:
            answers (List[str]): list of answers to build the option string
            option_type (str, optional): tells how to refer to the option. Defaults to "part".
            capitalize_option_type (bool, optional): indicates if the option type should be capitalized. Defaults to True.
            mode (int, optional): indicated version of the 1 answer string to use. Defaults to 1.
        """
        option_type = option_type.capitalize() if capitalize_option_type else option_type
        if isinstance(answers[0], str):
            if isinstance(option_type, list):
                option_type = option_type[0]
            if mode == 1:
                return f"{option_type} {answers[0]}"
            elif mode == 2:
                return f"Only {option_type} {answers[0]}"
            elif mode == 2:
                return f"{option_type} {answers[0]} is correct"
            elif mode == 3:
                return f"Only {option_type} {answers[0]} is correct"
            else:
                raise NotImplementedError(f"mode {mode} not implemented")
        else:
            assert len(answers) == 1, "answers should be of length 1"
            if isinstance(option_type, str):
                option_type = [option_type] * len(answers[0])
            
            if mode== 1:
                return ", ".join([f"{option_type[i]} {answers[0][i]}" for i in range(len(answers[0])-1)]) + \
                    f" and {option_type[-1]} {answers[0][-1]}"
            elif mode == 2:
                return ", ".join([f"{option_type[i]} {answers[0][i]}" for i in range(len(answers[0]))])
            else:
                raise NotImplementedError(f"mode {mode} not implemented for option_type {option_type}")
    
    @classmethod
    def single_answer_list_of_edges(
        self,
        answers: List[List[List[str]]],
        option_type: str = "part",
        capitalize_option_type: bool = True,
        mode: int = 1
    ):
        """create option string for single answer choice

        Args:
            answers (List[str]): list of answers to build the option string
            option_type (str, optional): tells how to refer to the option. Defaults to "part".
            capitalize_option_type (bool, optional): indicates if the option type should be capitalized. Defaults to True.
            mode (int, optional): indicated version of the 1 answer string to use. Defaults to 1.
        """
        # print("in opt func", answers)
        if capitalize_option_type:
            option_type = option_type.capitalize()
        return ", ".join([f"{option_type} {answers[0][i][0]} and {option_type} {answers[0][i][1]}" for i in range(len(answers[0]))])

    @classmethod
    def single_answer_tracking(
        self,
        answers: List[List[str]],
        option_type: str = "part",
        capitalize_option_type: bool = True
    ):
        answer_str = ""
        if capitalize_option_type:
            option_type = option_type.capitalize()
        
        image_refer = "image"
        if capitalize_option_type:
            image_refer = image_refer.capitalize()
        
        answers = answers[0]
        # print("single", len(answers))
        for i in range(len(answers)):
            if i == len(answers) - 1 and len(answers) > 1:
                answer_str += f" and {option_type} {answers[i][0]} in {image_refer} A matches {option_type} {answers[i][1]} in {image_refer} B"
            elif len(answers) == 1:
                answer_str += f"{option_type} {answers[i][0]} in {image_refer} A matches {option_type} {answers[i][1]} in {image_refer} B"
            else:
                answer_str += f"{option_type} {answers[i][0]} in {image_refer} A matches {option_type} {answers[i][1]} in {image_refer} B, "
        
        return answer_str

    @classmethod
    def template(self, answers: List[str], option_type: str = "part", 
                 capitalize_option_type: bool = True, mode: int = 1,
                 question_type: str = None, template_type: str = None):
        """template for multiple answer choices in question generation

        Args:
            answers (List[str]): list of answers to build the option string
            option_type (str, optional): tells how to refer to the option. Defaults to "part".
            capitalize_option_type (bool, optional): indicates if the option type should be capitalized. Defaults to True.
            mode (int, optional): indicated version of the answer string to use. Defaults to 1.
        """
        if len(answers) == 0:
            raise ValueError("No answers provided.")
        
        # TODO (07/10) - I have not tested this code to see whether it works
        # with multiple answers with track_multi. It should, but I have not tested
        # it and it is on low priority since I don't seem to have any such questions
        # in the annotations at the moment
        if len(answers) == 1:
            if question_type == "tracking" and template_type == "track_multi":
                # print("in track_multi")
                return self.single_answer_tracking(
                    answers=answers, option_type=option_type, 
                    capitalize_option_type=capitalize_option_type)
            elif isinstance(answers, list) and isinstance(answers[0], list) and isinstance(answers[0][0], list):
                option_str = self.single_answer_list_of_edges(
                    answers=answers, option_type=option_type, 
                    capitalize_option_type=capitalize_option_type, mode=mode)
            else:
                option_str = self.single_answer(
                    answers, option_type, capitalize_option_type, mode)
        elif len(answers) == 2:
            option_str = self.two_answers(answers, option_type, capitalize_option_type, mode)
        else:
            option_str = self.more_than_two_answers(answers, option_type, capitalize_option_type, mode)
        # remove extra white spaces
        option_str = re.sub(r"\s+", " ", option_str)
        return option_str.strip()

def get_option_label_separator(
    option_label_sep: Optional[Literal["dot", "single_parent", "double_parent", "single_square", "double_square"]] = "dot"
):
    """get the option label separator based on the provided type

    Args:
        option_label_sep (Optional[Literal["dot", "single_parent", "double_parent", "single_square", "double_square"]], optional): type of separator. Defaults to "dot".

    Returns:
        Callable: function to add the separator to the option label
    """
    if option_label_sep == "dot":
        add_label_sep = lambda x: f"{x}."
    elif option_label_sep == "single_parent":
        add_label_sep = lambda x: f"{x})"
    elif option_label_sep == "double_parent":
        add_label_sep = lambda x: f"({x})"
    elif option_label_sep == "single_square":
        add_label_sep = lambda x: f"{x}]"
    elif option_label_sep == "double_square":
        add_label_sep = lambda x: f"[{x}]"
    else:
        raise NotImplementedError(f"option_label_sep {option_label_sep} not implemented")
    return add_label_sep

def remove_option_label_separator(
    option_label_sep: Optional[Literal["dot", "single_parent", "double_parent", "single_square", "double_square"]] = "dot"
):
    """remove the option label separator based on the provided type

    Args:
        option_label_sep (Optional[Literal["dot", "single_parent", "double_parent", "single_square", "double_square"]], optional): type of separator. Defaults to "dot".

    Returns:
        Callable: function to remove the separator from the option label
    """
    if option_label_sep == "dot":
        remove_label_sep = lambda x: x[:-1]
    elif option_label_sep == "single_parent":
        remove_label_sep = lambda x: x[:-1]
    elif option_label_sep == "double_parent":
        remove_label_sep = lambda x: x[1:-1]
    elif option_label_sep == "single_square":
        remove_label_sep = lambda x: x[:-1]
    elif option_label_sep == "double_square":
        remove_label_sep = lambda x: x[1:-1]
    else:
        raise NotImplementedError(f"option_label_sep {option_label_sep} not implemented")
    return remove_label_sep


def form_question_string(
    question_template: str,
    correct_answers: List[str],
    incorrect_answers: List[str],
    named_params_for_question: Optional[Dict[str, Any]] = None,
    question_type: Optional[str] = None,
    question_template_type: Optional[str] = None,
    option_type: Optional[Union[str, List[str]]] = "part",
    capitalize_option_type: Optional[bool] = True,
    yes_and_no_question: Optional[bool] = False,
    use_furniture_annot: Optional[bool] = False,
    option_preprompt: Optional[str] = "\nOptions:\n",
    option_sep: Optional[str] = "\n",
    option_label_type: Optional[Literal["digit", "alpha_upper", "alpha_lower"]] = "alpha_upper",
    option_label_sep: Optional[Literal["dot", "single_parent", "double_parent", "single_square", "double_square"]] = "dot",
    add_abstain: Optional[bool] = False,
    add_none_at_end: Optional[bool] = True,
    randomize_options: Optional[bool] = False,
    single_answer_type: Optional[str] = str,
    num_options: Optional[int] = 4,
    furniture_annot: Optional[Dict[str, Any]] = None,
):
    """utility function to form the question string and the answer string

    Have tried to align pre-sets with VSI-Bench.
    
    NOTE: for now, we are assuming that correct_answers and incorrect_answers are
    lists of the same object/python type. will make this more generic later
    if needed.

    Args:
        question_template (str): template for the question
        correct_answers (List[str]): list of correct answers (raw). eg. list of part IDs that are correct
        incorrect_answers (List[str]): list of incorrect answers (raw). eg. list of part IDs that are incorrect
        named_params_for_question (Optional[Dict[str, Any]], optional): named parameters for filling blanks in the question. Defaults to None.
        question_type (Optional[str], optional): question type. Defaults to None.
        question_template_type (Optional[str], optional): question template type. Defaults to None.
        option_type (Optional[Union[str, List[str]]], optional): how to refer to each option (e.g., "part", "option", etc.). Defaults to "part".
            If a list is provided, it should be of the same length as the number of elements in each option.
            Each index of the list will be used to refer to the corresponding option.
            For example, if option_type = ["part", "option"], then the first option will be referred to as "part 1" and the second as "option 2".
            If the part is a list but the option type is a string, then the option type will be used to refer to all options.
        capitalize_option_type (Optional[bool], optional): capitalize the option type when listing the answer. Defaults to True.
        yes_and_no_question (Optional[bool], optional): . Defaults to False.
        use_furniture_annot (Optional[bool], optional): use furniture annotation to get the option text. Defaults to False.
        option_preprompt (Optional[str], optional): prepending string for the options. Defaults to "\nOptions:\n".
        option_label_type (Optional[Literal["digit", "alpha_upper", "alpha_lower"]], optional): type of label for the options. Defaults to "alpha_upper".
        option_label_sep (Optional[Literal["dot", "single_parent", "double_parent", "single_square", "double_square"]], optional): separator for the option label. Defaults to "dot".
        option_sep (Optional[str], optional): separator for the options, added to the end of the option string. Defaults to "\n".
        add_abstain (Optional[bool], optional): add an abstain option. Defaults to False.
        add_none_at_end (Optional[bool], optional): if adding a "none of the options" option, add it at the end, before abstaining label. Defaults to False.
        randomize_options (Optional[bool], optional): randomize the order and number of correct options. Defaults to False.
        single_answer_type (Optional[str]): type of single answer. can be str, or list or listoflists
        num_options (Optinal[int]): number of options. defaults to 4.
        furniture_annot Optional[Dict[str, Any]]: annotation for the part names of the furniture
    """
    
    # if we are to name the parts using furniture annotation, then do not 
    # name the option with any prefix like "part"
    if use_furniture_annot:
        option_type = ""
        for k, v in named_params_for_question.items():
            if v in furniture_annot["annotated_semantics"]:
                named_params_for_question[k] = furniture_annot["annotated_semantics"][v]
        if not yes_and_no_question:
            correct_answers = [furniture_annot["annotated_semantics"][ans] if ans in furniture_annot["annotated_semantics"] else ans for ans in correct_answers]
            incorrect_answers = [furniture_annot["annotated_semantics"][ans] if ans in furniture_annot["annotated_semantics"] else ans for ans in incorrect_answers]
        
    # fixing this for now. we expect a single answer option to be
    # a string. will worry about making this more generic later
    # if needed
    
    assert len(correct_answers) + len(incorrect_answers) >= 2, \
        f"len(correct_answers) {len(correct_answers)} + len(incorrect_answers) {len(incorrect_answers)} < 2"
    
    option_label_list = None
    if option_label_type == "digit":
        option_label_list = string.digits
    elif option_label_type == "alpha_upper":
        option_label_list = string.ascii_uppercase
    elif option_label_type == "alpha_lower":
        option_label_list = string.ascii_lowercase
    else:
        raise NotImplementedError(f"option_label_type {option_label_type} not implemented")
    # import ipdb; ipdb.set_trace()
    add_sep_to_opt_label = get_option_label_separator(option_label_sep)
    
    if named_params_for_question is None:
        named_params_for_question = {}
    
    # NOTE: might need to ammend this for specific question types 
    #     and question template types    
    filled_qstr = question_template.format(**named_params_for_question)
    qstr_without_opts = copy.deepcopy(filled_qstr)
    
    
    if option_preprompt is not None:
        filled_qstr += option_preprompt
    
    correct_option_raw = None # option text, as it is sourced from the correct_answers list
    correct_option_idx = None # index of the correct option as it is added to the question string
    correct_option_label = None # only the label of the correct option
    correct_option_text = None # text of the correct option, without label and separator
    correct_option_full_text = None # text of the correct option, with label and separator
    
    option_map = dict()
    
    if yes_and_no_question:
        num_options = 2
        all_options = [correct_answers[0].capitalize(), incorrect_answers[0].capitalize()]
        random.shuffle(all_options)
        for opt_idx, opt in enumerate(all_options):
            
            opt_text = answer_option_templates.template(
                [opt],
                option_type="",
                capitalize_option_type=capitalize_option_type,
                mode=1,
                question_type=question_type,
                template_type=question_template_type,
            )
            option_label = option_label_list[opt_idx]
            option_text = add_sep_to_opt_label(option_label) + " " + opt_text
            # print(correct_answers, incorrect_answers, opt, opt_text, option_text)
            if opt.lower() in [ans.lower() for ans in correct_answers]:
                correct_option_raw = opt
                correct_option_idx = opt_idx
                correct_option_label = option_label
                correct_option_text = opt_text
                correct_option_full_text = option_text
                # print(correct_option_label)
            
            option_map[opt_idx] = {
                "raw": opt,
                "label": option_label,
                "full_text": option_text,
                "text": opt_text,
            }
            filled_qstr += option_text + option_sep
        
        if add_abstain:
            abstain_opt = answer_option_templates.abstain_str(mode=1)
            option_label = option_label_list[num_options]
            option_text = add_sep_to_opt_label(option_label) + " " + abstain_opt
            option_map[num_options] = {
                "raw": ABSTAIN_RAW,
                "label": option_label,
                "text": abstain_opt,
                "full_text": option_text,
            }
            filled_qstr += option_text + option_sep
            num_options += 1
            
        return {
            "raw_qstr": qstr_without_opts,
            "qstr": filled_qstr,
            "options": option_map,
            "correct_option": {
                "raw": correct_option_raw,
                "label": correct_option_label,
                "text": correct_option_text,
                "idx": correct_option_idx,
                "full_text": correct_option_full_text,
            },
            "num_options": num_options,
        }
    
    num_correct_in_opt, num_incorrect_in_opt = 0, 0
    correct_in_opt_idxs, incorrect_in_opt_idxs = None, None
    
    if len(correct_answers) == 0:
        
        num_correct_in_opt = 0
        if min(len(incorrect_answers), num_options) > 2:
            num_incorrect_in_opt = int(np.random.randint(
                2, min(len(incorrect_answers), num_options)
            )) # max value is num_options - 1 
        else:
            num_incorrect_in_opt = 2
        assert num_incorrect_in_opt < num_options, \
            f"num_incorrect_in_opt {num_incorrect_in_opt} >= num_options {num_options} when num_correct_in_opt == 0"
        
        correct_in_opt_idxs = []
        incorrect_in_opt_idxs = np.random.choice(
            len(incorrect_answers), 
            num_incorrect_in_opt, 
            replace=False
        )
        
    elif len(correct_answers) == 1:
        
        num_correct_in_opt = 1
        num_incorrect_in_opt = len(incorrect_answers)
        if randomize_options:
            if len(incorrect_answers) > 1:
                num_incorrect_in_opt = int(np.random.randint(
                    1, min(len(incorrect_answers), num_options)))
        else:
            num_incorrect_in_opt = min(num_options - 1, len(incorrect_answers))
       
        assert num_correct_in_opt + num_incorrect_in_opt <= num_options, \
            f"num_correct_in_opt {num_correct_in_opt} + num_incorrect_in_opt {num_incorrect_in_opt} > num_options {num_options}"
        assert num_correct_in_opt + num_incorrect_in_opt >= num_options - 2, \
            f"num_correct_in_opt {num_correct_in_opt} + num_incorrect_in_opt {num_incorrect_in_opt} <= 0"
        
        correct_in_opt_idxs = [0]
        incorrect_in_opt_idxs = np.random.choice(
            len(incorrect_answers), 
            num_incorrect_in_opt, 
            replace=False
        )
    
    elif len(correct_answers) > 1:
        
        try:
            if randomize_options:
                if max(num_options-2-len(incorrect_answers), 0) == min(len(correct_answers), num_options):
                    num_correct_in_opt = min(len(correct_answers), num_options)
                else:
                    num_correct_in_opt = int(np.random.randint(
                        max(num_options-2-len(incorrect_answers), 0),
                        min(len(correct_answers), num_options)
                    ))
            else:
                num_correct_in_opt = min(len(correct_answers), num_options - 1)
        except:
            import ipdb; ipdb.set_trace()
        num_incorrect_in_opt = num_options - num_correct_in_opt
        if num_correct_in_opt != 1:
            num_incorrect_in_opt -= 1
        num_incorrect_in_opt = max(0, num_incorrect_in_opt)
        num_incorrect_in_opt = min(num_incorrect_in_opt, len(incorrect_answers))
        
        assert num_correct_in_opt + num_incorrect_in_opt <= num_options, \
            f"num_correct_in_opt {num_correct_in_opt} + num_incorrect_in_opt {num_incorrect_in_opt} > num_options {num_options}"
        
        try:
            assert num_correct_in_opt + num_incorrect_in_opt >= num_options - 2, \
                f"num_correct_in_opt {num_correct_in_opt} + num_incorrect_in_opt {num_incorrect_in_opt} <= num_options {num_options - 2}" # need at least 2 options
        except:
            import ipdb; ipdb.set_trace()
            
        if num_correct_in_opt == 0:
            try:
                assert num_incorrect_in_opt > 1 and num_incorrect_in_opt <= num_options - 1, \
                    f"num_incorrect_in_opt {num_incorrect_in_opt} not in [2, {num_options - 1}] when {num_correct_in_opt} == 0"
            except:
                import ipdb; ipdb.set_trace()
        if num_correct_in_opt > 1:
            assert num_correct_in_opt + num_incorrect_in_opt <= num_options - 1, \
                f"num_correct_in_opt {num_correct_in_opt} + num_incorrect_in_opt {num_incorrect_in_opt} > num_options - 1 {num_options - 1} when {num_correct_in_opt} > 1"

        correct_in_opt_idxs = np.random.choice(
            len(correct_answers), 
            num_correct_in_opt, 
            replace=False
        )
        incorrect_in_opt_idxs = np.random.choice(
            len(incorrect_answers), 
            num_incorrect_in_opt, 
            replace=False
        )
        
    correct_in_opt = [correct_answers[i] for i in correct_in_opt_idxs]
    incorrect_in_opt = [incorrect_answers[i] for i in incorrect_in_opt_idxs]
    
    all_single_options = correct_in_opt + incorrect_in_opt
    # print(all_single_options)
    is_correct_single_opt = [True] * len(correct_in_opt) + [False] * len(incorrect_in_opt)
    shuff_idxs = np.random.permutation(len(all_single_options))
    all_single_options = [all_single_options[i] for i in shuff_idxs]
    is_correct_single_opt = [is_correct_single_opt[i] for i in shuff_idxs]
    
    # NOTE: are the single answer options always of the single_answer_type?
    #       if not, we need to handle this case differently for multiple-
    #       answer options, hence setting this flag
    if single_answer_type == "listoflist":
        single_answer_type = str
    is_ans_sa_type = isinstance(all_single_options[0], single_answer_type)
    assert all([isinstance(opt, type(all_single_options[0])) for opt in all_single_options]), \
        f"all_single_options {all_single_options} not of the same type"
    
    option_map = dict()
    
    num_remaining_opts = num_options - len(all_single_options)
    single_correct_in_opt_labels, single_incorrect_in_opt_labels = None, None
    if num_remaining_opts > 0:
        # contains list of labels for single answer correct and incorrect options
        # to form the remaining options later for cases where the single answers
        # are not strings (e.g. denote a tuple of parts or an edge in the part graph)
        single_correct_in_opt_labels = list()
        single_incorrect_in_opt_labels = list()

    # first compile the single answer options
    for opt_idx in range(len(all_single_options)):
        
        opt = all_single_options[opt_idx]
        is_correct = is_correct_single_opt[opt_idx]
        # print((question_type == "temporal_loc" and question_template_type not in ["1part_order", "1part_order_last"]))
        # print(opt)
        opt_text = answer_option_templates.template(
            [opt],
            option_type=option_type,
            capitalize_option_type=capitalize_option_type,
            mode=2 if (question_type in ["temporal_loc", "temporal_ord"] and question_template_type not in ["1part_order", "1part_order_last", "latest_change", "next_change"]) else 1,
            question_type=question_type,
            template_type=question_template_type,
        )
        option_label = option_label_list[opt_idx]
        option_text = add_sep_to_opt_label(option_label) + " " + opt_text
            
        if is_correct and len(correct_in_opt) == 1:
            correct_option_raw = opt
            correct_option_idx = opt_idx
            correct_option_label = option_label
            correct_option_text = opt_text
            correct_option_full_text = option_text
        
        option_map[opt_idx] = {
            "raw": opt,
            "label": option_label,
            "text": opt_text,
            "full_text": option_text,
        }
        
        if num_remaining_opts > 0:
            if is_correct:
                single_correct_in_opt_labels.append(option_label)
            else:
                single_incorrect_in_opt_labels.append(option_label)
        
        filled_qstr += option_text + option_sep
        
    # num_remaining_opts = num_options - len(option_map)
    assert len(option_map) == len(all_single_options)
    assert num_remaining_opts <= num_options - 2, \
            f"num_remaining_opts {num_remaining_opts} > num_options - 2 {num_options - 2}"
    
    cur_opt_idx = len(option_map)
    remaining_opt_text = list()  
    remaining_opt_raw = list()
    remaining_opt_is_correct = list()
    
    if num_remaining_opts > 0:
        if len(correct_in_opt) > 1:
            remaining_opt_text.append(
                answer_option_templates.template(
                    correct_in_opt if is_ans_sa_type \
                        else single_correct_in_opt_labels,
                    option_type=option_type \
                        if is_ans_sa_type else "option",
                    capitalize_option_type=capitalize_option_type,
                    mode=1,
                    question_type=question_type,
                    template_type=question_template_type,
                )
            )
            remaining_opt_is_correct.append(True)
            remaining_opt_raw.append(correct_in_opt)
            
            if num_remaining_opts > 1:
                remaining_opt_text.append(
                    answer_option_templates.none_str(mode=1)
                )
                remaining_opt_raw.append(NONE_RAW)
                remaining_opt_is_correct.append(False)
                
        elif len(correct_in_opt) == 0:
            remaining_opt_text.append(
                answer_option_templates.none_str(mode=1)
            )
            remaining_opt_is_correct.append(True)
            remaining_opt_raw.append(NONE_RAW)
            
            if num_remaining_opts > 1:
                remaining_opt_text.append(
                    answer_option_templates.template(
                        incorrect_in_opt if is_ans_sa_type \
                            else single_incorrect_in_opt_labels,
                        option_type=option_type \
                            if is_ans_sa_type else "option",
                        capitalize_option_type=capitalize_option_type,
                        mode=1,
                        question_type=question_type,
                        template_type=question_template_type,
                    )
                )
                remaining_opt_raw.append(incorrect_in_opt)
                remaining_opt_is_correct.append(False)
        else:
            
            # TODO: if possible also handle the case of two incorrect answers here
            opt_text1 = answer_option_templates.template(
                [correct_in_opt[0], incorrect_in_opt[0]] \
                    if is_ans_sa_type \
                        else [single_correct_in_opt_labels[0], single_incorrect_in_opt_labels[0]],
                option_type=option_type \
                    if is_ans_sa_type else "option",
                capitalize_option_type=capitalize_option_type,
                mode=1,
                question_type=question_type,
                template_type=question_template_type,
            )
            opt_text2 = answer_option_templates.none_str(mode=1)
            remaining_opt_text = [opt_text1, opt_text2]
            remaining_opt_is_correct = [False, False]
            remaining_opt_raw = [[correct_in_opt[0], incorrect_in_opt[0]], NONE_RAW]
            
            if num_remaining_opts == 1:
                # random.shuffle(remaining_opt_text)
                if question_type in ["temporal_loc", "temporal_ord"]:
                    # for temporal ordering `temporal_loc`, always add the none option 
                    # if only one correct option is present and only one
                    # option is remaining
                    rdm_idx = 1
                else:
                    rdm_idx = np.random.randint(0, 2)
                # NOTE: no need to shuffle the is_correct list
                # because both options are incorrect
                remaining_opt_text = [remaining_opt_text[rdm_idx]]  
                remaining_opt_is_correct = [remaining_opt_is_correct[rdm_idx]]
                remaining_opt_raw = [remaining_opt_raw[rdm_idx]]
    
    # NOTE: at this point, all remaining options are accounted for    
    
    none_idx = None # index of the none option
    if add_none_at_end:
        none_str = answer_option_templates.none_str(mode=1)
        for idx in range(num_remaining_opts):
            if remaining_opt_text[idx] == none_str:
                none_idx = idx
                break
        if none_idx is not None:
            # pop the none option and append at the end
            none_str = remaining_opt_text.pop(none_idx)
            none_is_correct = remaining_opt_is_correct.pop(none_idx)
            none_raw = remaining_opt_raw.pop(none_idx)
            remaining_opt_text.append(none_str)
            remaining_opt_is_correct.append(none_is_correct)
            remaining_opt_raw.append(none_raw)
        
    else:
        # shuffle the remaining options
        shuff_idxs = np.random.permutation(num_remaining_opts)
        remaining_opt_text = [remaining_opt_text[i] for i in shuff_idxs]
        remaining_opt_is_correct = [remaining_opt_is_correct[i] for i in shuff_idxs]
        remaining_opt_raw = [remaining_opt_raw[i] for i in shuff_idxs]
        
    for remaining_opt_idx in range(cur_opt_idx, cur_opt_idx + num_remaining_opts):
        opt_text = remaining_opt_text[remaining_opt_idx - cur_opt_idx]                
        is_correct = remaining_opt_is_correct[remaining_opt_idx - cur_opt_idx]
        opt_raw = remaining_opt_raw[remaining_opt_idx - cur_opt_idx]
        
        option_label = option_label_list[remaining_opt_idx]
        option_text = add_sep_to_opt_label(option_label) + " " + opt_text
        
        if is_correct:
            correct_option_raw = opt_raw
            correct_option_idx = remaining_opt_idx
            correct_option_label = option_label
            correct_option_text = opt_text
            correct_option_full_text = option_text
            
        
        option_map[remaining_opt_idx] = {
            "raw": opt_raw,
            "label": option_label,
            "text": opt_text,
            "full_text": option_text,
        }
        filled_qstr += option_text + option_sep        
    
    if add_abstain:
        abstain_opt = answer_option_templates.abstain_str(mode=1)
        option_label = option_label_list[num_options]
        option_text = add_sep_to_opt_label(option_label) + " " + abstain_opt
        option_map[num_options] = {
            "raw": ABSTAIN_RAW,
            "label": option_label,
            "text": abstain_opt,
            "full_text": option_text,
        }
        filled_qstr += option_text + option_sep
        num_options += 1

    return {
        "raw_qstr": qstr_without_opts,
        "qstr": filled_qstr,
        "options": option_map,
        "correct_option": {
            "raw": correct_option_raw,
            "label": correct_option_label,
            "text": correct_option_text,
            "idx": correct_option_idx,
            "full_text": correct_option_full_text,
        },
        "num_options": num_options,
    }
    
def check_question_answer_string(
    question_dict: Dict[str, Any],
    num_options: int,
    correct_answers: List[str],
    incorrect_answers: List[str],
    option_sep: Optional[str] = "\n",
    option_preprompt: Optional[str] = "\nOptions:\n",
    option_label_sep: Optional[Literal["dot", "single_parent", "double_parent", "single_square", "double_square"]] = "dot",
    template_specific_check: Optional[List[Callable]]=None,
    **kwargs
):
    """question validity checker for the generated question and answer strings

    Args:
        question_dict (Dict[str, Any]): question dict as returned by the form_question_string function
        num_options (int): total number of options in the question
        correct_answers (List[str]): list of raw correct answers as provided by the template
        incorrect_answers (List[str]): list of raw incorrect answers as provided by the template
        option_sep (Optional[str], optional): option separator. Defaults to "\n".
        option_preprompt (_type_, optional): pre-prompt used in the question string. Defaults to "\nOptions:\n".
        template_specific_check (Optional[List[Callable]], optional): any template-specific checks. Defaults to None.

    Raises:
        ValueError: _description_
        
    Returns:
        bool: True if the question and answer strings are valid, else will raise error
    """
    
    # flag to check if single-answer options are strings
    is_sa_opt_str = isinstance(correct_answers[0] if correct_answers else incorrect_answers[0], str)
    
    add_sep_to_opt_label = get_option_label_separator(option_label_sep)
    
    # check if number of optionsis correct
    completed_question_string = question_dict["qstr"]
    options_list_string = completed_question_string.split(option_preprompt)[-1]
    options_string_list = options_list_string.split(option_sep)
    options_string_list = [opt.strip() for opt in options_string_list if opt.strip()]
    detected_num_opts = len(options_string_list)
    
    if detected_num_opts != num_options:
        raise ValueError(f"detected_num_opts {detected_num_opts} != num_options {num_options}")
    
    # check that correct ans is correct and incorrect ans is incorrect
    options = question_dict["options"]
    
    correct_option_data = question_dict["correct_option"]
    correct_option_raw = correct_option_data["raw"]
    correct_option_label = correct_option_data["label"]
    correct_option_idx = correct_option_data["idx"]
    correct_option_full_text = correct_option_data["full_text"]
    
    # check that correct option is not abstrain
    if correct_option_raw == ABSTAIN_RAW:
        raise ValueError(f"correct_option_raw {correct_option_raw} == ABSTAIN_RAW {ABSTAIN_RAW}")
    
    # check that the raw options are stored correctly in the options
    assert correct_option_raw == options[correct_option_idx]["raw"], \
        f"correct_option_raw {correct_option_raw} != options[correct_option_idx]['raw'] {options[correct_option_idx]['raw']}"
    
    # check the option labels are stored correctly in the options
    assert correct_option_label == options[correct_option_idx]["label"], \
        f"correct_option_label {correct_option_label} != options[correct_option_idx]['label'] {options[correct_option_idx]['label']}"
    
    # check that the option label text is correct in the string
    assert add_sep_to_opt_label(correct_option_label) == options_string_list[correct_option_idx].split(" ")[0], \
        f"add_sep_to_opt_label(correct_option_label) {add_sep_to_opt_label(correct_option_label)} \
            != options_string_list[correct_option_idx].split(' ')[0] {options_string_list[correct_option_idx].split(' ')[0]}"
    
    # 
    assert f"{correct_option_full_text}" == options_string_list[correct_option_idx], \
        f"correct_option_full_text {correct_option_full_text} != options_string_list[correct_option_idx] {options_string_list[correct_option_idx]}"

    # check that all raw answers in the incorrect options are actually incorrect  
    # also check that if there are more than one correct answers, then they are
    # not marked as correct_option_idx
    # TODO: need to fix this for `find_edge` template
    cnt_incorrect_opts = 0
    cnt_correct_opts = 0
    for opt_idx in range(num_options):
        if opt_idx == correct_option_idx:
            continue
        elif options[opt_idx]["raw"] in [ABSTAIN_RAW, NONE_RAW]:
            continue
        
        opt_raw = options[opt_idx]["raw"]
        # NOTE: we implement this check for `find_edge` this way
        # because the options can also have NONE_RAW as the raw
        # answer which is not a list
        if (not isinstance(opt_raw, list)) or \
            (isinstance(opt_raw, list) and isinstance(opt_raw[0], str)):
            opt_raw = [opt_raw]            
        
        if all([opt in correct_answers for opt in opt_raw]):
            cnt_correct_opts += 1
        else:
            cnt_incorrect_opts += 1
    
    # if you count num_options -1 number of incorrect options, then
    #    the last option is none of the above
    # if cnt_incorrect_opts == num_options-1 and (isinstance(correct_option_raw, list) and len(correct_option_raw) > 1):
    if cnt_incorrect_opts == num_options-1:
        assert cnt_correct_opts == 0, \
                f"#correct answers in options apart from correct option {cnt_correct_opts} != 0 \
                    when #incorrect answers in options {cnt_incorrect_opts} == num_options-1 {num_options-1}"
            
        if cnt_correct_opts == 0:
            try:
                check_none_or_single_correct_option = lambda x: \
                    ((x == NONE_RAW) or \
                        ((is_sa_opt_str and isinstance(x, str)) or \
                        (not is_sa_opt_str and isinstance(x[0], str)) \
                            and (x in correct_answers))
                    )
                assert check_none_or_single_correct_option(correct_option_raw), \
                    f"correct_option_raw {correct_option_raw} is not a single correct option or NONE_RAW {NONE_RAW}"
                assert check_none_or_single_correct_option(options[correct_option_idx]["raw"]), \
                    f"options[correct_option_idx]['raw'] {options[correct_option_idx]['raw']} is not a single correct option or NONE_RAW {NONE_RAW}"
                # assert options[correct_option_idx]["raw"] == NONE_RAW, \
                #     f"options[correct_option_idx]['raw'] {options[correct_option_idx]['raw']} != NONE_RAW {NONE_RAW}"
                # assert correct_option_raw == NONE_RAW, \
                #     f"correct_option_raw {correct_option_raw} != NONE_RAW {NONE_RAW}"
                
            except:
                import ipdb; ipdb.set_trace()
        
    # if number of correct answers apart from the correct option
    # is 1, this is wrong as in case of multiple correct answers, there
    # should be at least 2 correct answers apart from the correct option
    if cnt_correct_opts == 1:
        raise ValueError(f"cnt_correct_opts {cnt_correct_opts} == 1")
        
    
    # check that all raw answers in the correct option are actually correct
    if correct_option_raw != NONE_RAW:
        # import ipdb; ipdb.set_trace()
        flatten = False
        if (not isinstance(correct_option_raw, list)) or ((not is_sa_opt_str) and isinstance(correct_option_raw[0], str)):
            correct_option_raw = [correct_option_raw]
            flatten = True
        assert all([opt in correct_answers for opt in correct_option_raw]), \
            f"correct_option_raw {correct_option_raw} not in correct_answers {correct_answers}. not all correct options are actually correct"
        if flatten:
            correct_option_raw = correct_option_raw[0]
            
        
    if template_specific_check is not None:
        for check in template_specific_check:
            assert check(question_dict, num_options,
                         correct_answers,
                         incorrect_answers,
                         option_sep=option_sep,
                         option_preprompt=option_preprompt,
                         option_label_sep=option_label_sep,
                         **kwargs), f"template_specific_check {check} failed"
    
    return True
    