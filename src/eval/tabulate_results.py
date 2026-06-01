"""Tabulate Flat-Pack Bench results and bootstrap confidence intervals."""

import pyrootutils

root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import argparse
import json
import os
import os.path as osp
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.eval.evaluate import evaluate, post_process_registry


RESULT_COLUMNS = ["micro_avg", "temporal_ord", "temporal_loc", "tracking", "mating"]


def hierarchical_bootstrap_video_only(
    video_id: np.ndarray,
    human_correct: np.ndarray,
    model_correct: np.ndarray,
    n_videos_sample: int = 50,
    bootstrap_iters: int = 100_000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    video_id = np.asarray(video_id)
    human_correct = np.asarray(human_correct)
    model_correct = np.asarray(model_correct)

    videos = np.unique(video_id)
    human_by_video = []
    model_by_video = []
    for video in videos:
        mask = video_id == video
        human_by_video.append(human_correct[mask])
        model_by_video.append(model_correct[mask])

    gaps = np.empty(bootstrap_iters, dtype=float)
    human_trials = np.empty(bootstrap_iters, dtype=float)
    model_trials = np.empty(bootstrap_iters, dtype=float)

    for trial_idx in range(bootstrap_iters):
        sampled_video_idx = rng.integers(0, len(videos), size=n_videos_sample)
        human_sum = 0.0
        model_sum = 0.0
        total_questions = 0

        for video_idx in sampled_video_idx:
            human_sum += human_by_video[video_idx].sum()
            model_sum += model_by_video[video_idx].sum()
            total_questions += len(human_by_video[video_idx])

        gaps[trial_idx] = (human_sum - model_sum) / total_questions
        human_trials[trial_idx] = human_sum / total_questions
        model_trials[trial_idx] = model_sum / total_questions

    return gaps, human_trials, model_trials


def post_process_response(response: dict, model_key: str) -> Optional[str]:
    if model_key == "gemini":
        return response.get("post_processed_response")
    return post_process_registry[model_key](response["response"])


def get_confidence_intervals(
    responses_fn: str,
    model_key: str,
    n_videos_sample: int = 50,
    bootstrap_iters: int = 100_000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    predictions = []
    labels = []
    video_ids = []

    with open(responses_fn) as f:
        for line in f:
            response = json.loads(line)
            predictions.append(post_process_response(response, model_key))
            labels.append(response["question"]["question"]["correct_option"]["label"])
            video_ids.append(response["question"]["video_id"])

    video_id_to_idx = {video_id: idx for idx, video_id in enumerate(sorted(set(video_ids)))}
    video_idx = np.asarray([video_id_to_idx[video_id] for video_id in video_ids], dtype=int)
    model_correct = np.asarray(
        [int(prediction == label) for prediction, label in zip(predictions, labels)],
        dtype=float,
    )

    _, _, model_trials = hierarchical_bootstrap_video_only(
        video_id=video_idx,
        human_correct=np.zeros_like(model_correct),
        model_correct=model_correct,
        n_videos_sample=n_videos_sample,
        bootstrap_iters=bootstrap_iters,
        seed=seed,
    )
    ci_low, ci_high = np.percentile(model_trials, [2.5, 97.5])
    return float(ci_low * 100), float(model_trials.mean() * 100), float(ci_high * 100)


def infer_result_metadata(result_dir: str, config_path: str) -> Tuple[str, str, str, int]:
    result_name = osp.basename(result_dir)

    if "videorefer" in result_name:
        return "videorefer", "videorefer", "videorefer_videorefer", 1

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_config = config.get("model", {})
    model_name = model_config.get("model_name", result_name)
    model_target = model_config.get("_target_", "")
    model_key = model_target.split(".")[-2]
    setting = osp.basename(config.get("media_cache_dir", result_name))
    num_shuffles = int(config.get("num_shuffles", 1) or 1)

    return model_name, model_key, setting, num_shuffles


def extract_label(pattern: str, value: str, default: str = "unknown") -> str:
    match = re.search(pattern, value)
    return match.group() if match else default


def tabulate_results(
    responses_dir: str,
    bootstrap_iters: int = 100_000,
    seed: int = 42,
) -> pd.DataFrame:
    results = {}

    for result_dirname in sorted(os.listdir(responses_dir)):
        result_path = osp.join(responses_dir, result_dirname)
        config_path = osp.join(result_path, "config.yaml")
        responses_path = osp.join(result_path, "responses.jsonl")

        if not osp.isdir(result_path) or not osp.exists(responses_path):
            continue
        if "videorefer" not in result_dirname and not osp.exists(config_path):
            continue

        model_name, model_key, setting, num_shuffles = infer_result_metadata(
            result_path,
            config_path,
        )
        result_df = evaluate(
            responses_fn=responses_path,
            model_name=model_key,
            num_shuffs_to_consider=num_shuffles,
        )
        strict_row = result_df["strict"]
        ci_low, ci_mean, ci_high = get_confidence_intervals(
            responses_path,
            model_key,
            bootstrap_iters=bootstrap_iters,
            seed=seed,
        )

        row_key = f"{model_name}_{setting}"
        results[row_key] = {
            "Model": model_name,
            "Visual Prompt Type": extract_label(r"^(sep|concat|collage|videorefer)", setting),
            "Video Type": extract_label(r"(keyframe|trimmed|videorefer)", setting),
            **{column: strict_row[column] for column in RESULT_COLUMNS},
            "ci_low": ci_low,
            "ci_mean": ci_mean,
            "ci_high": ci_high,
        }

    results_df = pd.DataFrame.from_dict(results, orient="index")
    if results_df.empty:
        return results_df

    return results_df[
        ["Model", "Visual Prompt Type", "Video Type", *RESULT_COLUMNS, "ci_low", "ci_mean", "ci_high"]
    ].sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tabulate Flat-Pack Bench results with bootstrap CIs.")
    parser.add_argument("--responses-dir", default=osp.join(root, "src", "eval", "results", "responses"))
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--bootstrap-iters", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_df = tabulate_results(
        args.responses_dir,
        bootstrap_iters=args.bootstrap_iters,
        seed=args.seed,
    )

    if args.output_csv:
        results_df.to_csv(args.output_csv, index=False)
    print(results_df)


if __name__ == "__main__":
    main()
