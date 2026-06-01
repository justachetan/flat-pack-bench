import os
import os.path as osp
import json
import yaml
import glob
from collections import Counter


def evaluate(
    cache_dir_root: str,
    compare_with_response_jsonl: str = None,
):

    num_abstain = 0
    num_failed_code = 0
    num_correct = 0
    num_incorrect = 0
    num_total = 0

    cache_fns = sorted(
        os.listdir(cache_dir_root),
        key=lambda x: int(x.split(".")[0]),
    )

    print(f"Found {len(cache_fns)} question cache directories in {cache_dir_root}.")

    with open(compare_with_response_jsonl, "r") as f:
        cmpr_lines = f.readlines()
    cmpr_lines = [json.loads(line) for line in cmpr_lines]

    correct_video_ids = list()
    correct_question_categories = list()
    correct_furniture_names = list()
    
    intersection_with_lmm_idx = list()
    intersection_with_lmm_question_categories = list()
    intersection_with_lmm_furniture_names = list()
    intersection_with_lmm_video_ids = list()

    lmm_err_recovery_count = 0
    lmm_err_count = 0

    for idx, cache_fn in enumerate(cache_fns):
        
        execution_results_fn = osp.join(cache_dir_root, cache_fn, "execution_result.json")
        if osp.exists(execution_results_fn):
            
            with open(execution_results_fn, "r") as f:
                execution_results = json.load(f)
        else:
            print(f"Warning: execution_result.json not found for {cache_fn}, skipping.")
            num_failed_code += 1
            num_total += 1
            continue
        
        cmpr_entry = None
        for idx in range(len(cmpr_lines)):
            if cmpr_lines[idx]["question"]["qid_flat"].split("/")[-2] == cache_fn.split(".")[0]:
                cmpr_entry = cmpr_lines[idx]
                break
        if cmpr_entry is None:
            print("cache_fn:", cache_fn)
        lmm_answer = cmpr_entry.get("post_processed_response")
        # import ipdb; ipdb.set_trace()
        lmm_correct_option = cmpr_entry.get("question").get("question").get("correct_option").get("label")
        lmm_is_incorrect = lmm_answer != lmm_correct_option
        # import ipdb; ipdb.set_trace()
        if lmm_is_incorrect:
            lmm_err_count += 1


        if not execution_results["ok"]:
            num_failed_code += 1
        else:
            pred_result = execution_results["result"]
            question_fn = osp.join(cache_dir_root, cache_fn, "qinfo.json")
            if not osp.exists(question_fn):
                continue
            

            with open(question_fn, "r") as f:
                question_info = json.load(f)
            question = question_info["question"]
            
            video_id = question_info["video_id"]
            question_category = question_info["question_category"]

            abstained = False
            for opt_idx, opt in question["options"].items():
                if opt["raw"] == "abstain":
                    if pred_result == opt["label"]:
                        num_abstain += 1
                        abstained = True
                    break
                
            correct_option = question["correct_option"]
            

            if not abstained:
                if pred_result == correct_option["label"]:
                    num_correct += 1
                    correct_video_ids.append(video_id)
                    correct_question_categories.append(question_category)
                    correct_furniture_names.append(question_info.get("furniture_name", "N/A"))
                    
                    if not lmm_is_incorrect:
                        intersection_with_lmm_idx.append(idx)
                        intersection_with_lmm_question_categories.append(question_category)
                        intersection_with_lmm_furniture_names.append(question_info.get("furniture_name", "N/A"))
                        intersection_with_lmm_video_ids.append(video_id)
                    else:
                        lmm_err_recovery_count += 1
                     
                else:
                    num_incorrect += 1

       

                

        num_total += 1

    accuracy = num_correct / len(cache_fns) * 100
    abstain_rate = num_abstain / len(cache_fns) * 100
    fail_rate = num_failed_code / len(cache_fns) * 100
    lmm_err_recovery_rate = (lmm_err_recovery_count / lmm_err_count if lmm_err_count > 0 else 0) * 100
    lmm_accuracy = (len(cache_fns) - lmm_err_count) / len(cache_fns) * 100 if len(cache_fns) > 0 else 0

    return {
        "accuracy": accuracy,
        "abstain_rate": abstain_rate,
        "fail_rate": fail_rate,
        "num_total": num_total,
        "num_correct": num_correct,
        "num_abstain": num_abstain,
        "num_failed_code": num_failed_code,
        "num_incorrect": num_incorrect,
        "correct_video_ids": Counter(correct_video_ids),
        "correct_question_categories": Counter(correct_question_categories),
        "correct_furniture_names": Counter(correct_furniture_names),
        "intersection_with_lmm_idx": intersection_with_lmm_idx,
        "num_intersection_with_lmm": len(intersection_with_lmm_idx),
        "intersection_with_lmm_question_categories": Counter(intersection_with_lmm_question_categories),
        "intersection_with_lmm_furniture_names": Counter(intersection_with_lmm_furniture_names),
        "intersection_with_lmm_video_ids": Counter(intersection_with_lmm_video_ids),
        "lmm_err_recovery_count": lmm_err_recovery_count,
        "lmm_err_count": lmm_err_count,
        "lmm_err_recovery_rate": lmm_err_recovery_rate,
        "lmm_accuracy": lmm_accuracy,
    }

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache_dir_root",
        type=str,
        required=True,
        help="Path to the root cache directory containing all question subdirectories.",
    )
    parser.add_argument(
        "--compare_with_response_jsonl",
        type=str,
        required=True,
        help="Path to the JSONL file containing model responses to compare with.",
    )
    args = parser.parse_args()

    results = evaluate(
        cache_dir_root=args.cache_dir_root,
        compare_with_response_jsonl=args.compare_with_response_jsonl,
    )

    print("Evaluation Results:")
    for k, v in results.items():
        if isinstance(v, Counter):
            print(f"{k}:")
            for item, count in v.items():
                print(f"  {item}: {count}")
            print("  --------------------------")
            print("  total:", sum(v.values()))
        else:
            print(f"{k}: {v}")

    
                
            

