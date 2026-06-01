import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from typing import Iterable, List, Optional, Union
import json
from collections import Counter

import numpy as np
import pandas as pd


from src.eval.models.gemini import post_process_response as gemini_post_process
from src.eval.models.qwen_2_5_vl_hf import post_process_response as qwen2_5vl_post_process
from src.eval.models.qwen_3_vl_hf import post_process_response as qwen3vl_post_process
from src.eval.models.openai_gpt import post_process_response as openai_gpt_post_process
from src.eval.models.llava_next_video import post_process_response as llava_next_video_post_process
from src.eval.models.llava_ov import post_process_response as llava_ov_post_process
from src.eval.models.internvl3 import post_process_response as internvl3_post_process
from src.eval.models.video_llava import post_process_response as video_llava_post_process
from src.eval.models.arrowrl import post_process_response as arrowrl_post_process
# from src.eval.models.llava_video import post_process_response as llava_video_post_process
from src.eval.models.llava_next_video import post_process_response as llava_video_post_process
from src.eval.models.model_utils.perceptionlm.post_process import post_process_response as perceptionlm_post_process
from src.eval.models.aria import post_process_response as aria_post_process


def identity_post_process(response):
    return response


post_process_registry = {
    "gemini": gemini_post_process,
    "qwen_2_5_vl_hf": qwen2_5vl_post_process, 
    "qwen_2_vl_hf": qwen2_5vl_post_process,
    "qwen_3_vl_hf": qwen3vl_post_process,
    "openai_gpt": openai_gpt_post_process,
    "llava_next_video": llava_next_video_post_process,
    "llava_ov": llava_ov_post_process,
    "internvl3": internvl3_post_process,
    "video_llava": video_llava_post_process,
    "llava_video": llava_video_post_process,
    "arrowrl": arrowrl_post_process,
    "perceptionlm": perceptionlm_post_process,
    "genfs": gemini_post_process,
    "aria": aria_post_process,
    "videorefer": identity_post_process,
}


def _normalize_response_paths(
    responses_fn: Optional[str] = None,
    responses_fns: Optional[Union[str, Iterable[str]]] = None,
) -> List[str]:
    if (responses_fn is None) == (responses_fns is None):
        raise ValueError("Exactly one of responses_fn or responses_fns must be provided.")
    
    if responses_fn is not None:
        return [responses_fn]
    
    if isinstance(responses_fns, str):
        raw_paths = responses_fns.split(",")
    else:
        raw_paths = []
        for path in responses_fns:
            raw_paths.extend(str(path).split(","))
    
    response_paths = [path.strip() for path in raw_paths if path and path.strip()]
    if not response_paths:
        raise ValueError("responses_fns did not contain any response paths.")
    return response_paths


def evaluate(
    responses_fn: Optional[str] = None,
    responses_fns: Optional[Union[str, Iterable[str]]] = None,
    model_name: str = "gemini",
    num_shuffs_to_consider: int = None,
    binary_resp_acc: bool = False,
    verbose: bool = False,
):
    response_paths = _normalize_response_paths(responses_fn, responses_fns)
    per_question_cnt = {}
    per_category_cnt_major = {}
    per_category_cnt_strict = {}
    per_category_cnt_any = {}
    tot_category_cnt = {}
    
    per_category_opts_cnt = {}
    per_category_opts = {}
    per_category_opts_raw = {}
    tot_category_opts_raw = {}

    num_lines = 0
    for response_path in response_paths:
        with open(response_path) as f:
            for line in f:
                num_lines += 1
                conv_data = json.loads(line.strip())
                qid_flat = conv_data["question"]["qid_flat"]
                
                question_cat = conv_data["question"]["question_category"]
                
                question_number, shuff_idx = qid_flat.split("/")[-2:]

                if num_shuffs_to_consider is not None and int(shuff_idx) >= num_shuffs_to_consider:
                    continue
                
                unprocessed_response = conv_data["response"]
                
                if model_name == "gemini":
                    if isinstance(conv_data["response"], dict):
                        unprocessed_response = conv_data["response"].get("response", conv_data["response"])
                    pred_ans = conv_data["post_processed_response"]
                else:
                    pred_ans = post_process_registry[model_name](conv_data["response"])
                
                true_ans = conv_data["question"]["question"]["correct_option"]["label"]
                true_ans_raw = conv_data["question"]["question"]["correct_option"]["raw"]
                
                if binary_resp_acc:
                    true_ans_raw = true_ans_raw.lower()
                    if true_ans_raw not in per_category_opts_raw:
                        per_category_opts_raw[true_ans_raw] = 0
                    if true_ans_raw not in tot_category_opts_raw:
                        tot_category_opts_raw[true_ans_raw] = 0
                    question_cat = true_ans_raw
                    tot_category_opts_raw[true_ans_raw] += 1
                    
                if verbose:
                    print(
                        f"true_response: {true_ans}", 
                        f"unprocessed response: {unprocessed_response}", 
                        f"processed response: {pred_ans}", 
                        f"correct: {pred_ans == true_ans}"
                    )
                
                if question_number not in per_question_cnt:
                    per_question_cnt[question_number] = {
                        "correct": 0,
                        "incorrect": 0,
                        "category": question_cat,
                        "num_options": len(conv_data["question"]["question"]["options"]),
                        "true_ans": true_ans,
                    }
                    if question_cat not in tot_category_cnt:
                        tot_category_cnt[question_cat] = 0
                    tot_category_cnt[question_cat] += 1
                    if question_cat not in per_category_cnt_major:
                        per_category_cnt_major[question_cat] = 0
                    if question_cat not in per_category_cnt_strict:
                        per_category_cnt_strict[question_cat] = 0
                    if question_cat not in per_category_cnt_any:
                        per_category_cnt_any[question_cat] = 0
                    if question_cat not in per_category_opts_cnt:
                        per_category_opts_cnt[question_cat] = []
                    per_category_opts_cnt[question_cat].append(len(conv_data["question"]["question"]["options"]))
                    if question_cat not in per_category_opts:
                        per_category_opts[question_cat] = []
                    per_category_opts[question_cat].append(true_ans)
                
                if pred_ans is None:
                    pred_ans = ""
                
                if pred_ans == true_ans:
                    per_question_cnt[question_number]["correct"] += 1
                    if binary_resp_acc:
                        per_category_opts_raw[true_ans_raw] += 1
                else:
                    per_question_cnt[question_number]["incorrect"] += 1
                
    print("No. of lines:", num_lines)
    for question_number in per_question_cnt:
        category = per_question_cnt[question_number]["category"]
        if per_question_cnt[question_number]["correct"] > per_question_cnt[question_number]["incorrect"]:
            per_category_cnt_major[category] += 1
        if per_question_cnt[question_number]["incorrect"] == 0:
            per_category_cnt_strict[category] += 1
        if per_question_cnt[question_number]["correct"] > 0:
            per_category_cnt_any[category] += 1
    
    stats_dict = {mode: {category: 0 for category in tot_category_cnt} for mode in ["strict", "majority"]}
    stats_dict["strict"] = {category: per_category_cnt_strict[category] *100 / tot_category_cnt[category] for category in tot_category_cnt}
    stats_dict["strict"]["micro_avg"] = sum(list(per_category_cnt_strict.values())) * 100 / sum(list(tot_category_cnt.values()))

    stats_dict["majority"] = {category: per_category_cnt_major[category] * 100 / tot_category_cnt[category] for category in tot_category_cnt}       
    stats_dict["majority"]["micro_avg"] = sum(list(per_category_cnt_major.values())) * 100 / sum(list(tot_category_cnt.values()))

    stats_dict["any"] = {category: per_category_cnt_any[category] * 100 / tot_category_cnt[category] for category in tot_category_cnt}
    stats_dict["any"]["micro_avg"] = sum(list(per_category_cnt_any.values())) * 100 / sum(list(tot_category_cnt.values()))
    
    stats_dict["random_chance"] = {category: (100 / np.array(per_category_opts_cnt[category])).mean() for category in tot_category_cnt}
    stats_dict["freq_chance"] = {category: Counter(per_category_opts[category]).most_common(1)[0][1] * 100 / len(per_category_opts[category]) for category in tot_category_cnt}

    stats_dict["random_chance"]["micro_avg"] = (100 / np.array(sum(list(per_category_opts_cnt.values()), []))).mean()
    stats_dict["freq_chance"]["micro_avg"] = Counter(sum(list(per_category_opts.values()), 
                                                           [])).most_common(1)[0][1] * 100 / len(sum(list(per_category_opts.values()), []))

    print(tot_category_cnt)
    
    print(per_category_cnt_major)
    stats_dict = pd.DataFrame(stats_dict) 
    
    # Reorder result categories
    if not binary_resp_acc:
        desired_order = ["micro_avg", "temporal_ord", "temporal_loc", "tracking","mating"]
        stats_dict = stats_dict.loc[desired_order]
        
    else:
        desired_order = ["micro_avg", "yes", "no"]
        stats_dict = stats_dict.loc[desired_order]

    return stats_dict


def main():
    
    import argparse
    
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--responses_fn", help="Path to one response JSONL file.")
    input_group.add_argument("--responses_fns", nargs="+", help="Paths to response JSONL files. Comma-separated values are also accepted.")
    parser.add_argument("--model_name", type=str, choices=list(post_process_registry.keys()), default="gemini", help="Model name for evaluation. Determines the post-processing function to apply to model responses.")
    parser.add_argument("--num_shuffs_to_consider", type=int, default=1, help="(Legacy) Number of option shuffles to consider for evaluation. If None, considers all shuffles.")
    parser.add_argument("--binary_resp_acc", action="store_true", help="Whether to evaluate binary response accuracy.")

    args = parser.parse_args()

    stats_df = evaluate(
        responses_fn=args.responses_fn,
        responses_fns=args.responses_fns,
        model_name=args.model_name,
        num_shuffs_to_consider=args.num_shuffs_to_consider,
        binary_resp_acc=args.binary_resp_acc,
    )
    response_input = args.responses_fn if args.responses_fn is not None else args.responses_fns
    print(f"Evaluation results for {args.model_name} on {response_input}:")
    
    print(stats_df.T.to_csv())
    
if __name__ == "__main__":
    main()
