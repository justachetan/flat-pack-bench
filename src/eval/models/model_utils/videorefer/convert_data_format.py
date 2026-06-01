from __future__ import annotations
import pyrootutils
ROOT = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import argparse
import glob
import json
import os
import os.path as osp
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import imageio.v2 as iio
import numpy as np
import yaml
from pycocotools.mask import decode, encode
from tqdm import tqdm

from src.eval.prompts.templates.questions.convert_yaml_to_json import (
    QUESTION_TEMPLATE_REGISTRY,
)

def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def _sorted_question_yaml_files(question_dir: str) -> list[str]:
    return sorted(
        glob.glob(osp.join(question_dir, "*.yaml")),
        key=lambda x: int(osp.basename(x).split(".")[0]),
    )


def _zero_mask_like(prompt_masks_for_frame: dict[str, Any]) -> dict[str, Any]:
    if not prompt_masks_for_frame:
        raise ValueError("Cannot create an empty fallback mask without a reference mask.")

    first_mask = next(iter(prompt_masks_for_frame.values()))
    mask_rle = encode(np.asfortranarray(np.zeros_like(decode(first_mask))))
    mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
    return mask_rle


def _replace_part_mentions(raw_qstr: str, replacements: list[tuple[str, str]]) -> str:
    for part_id, object_token in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        raw_qstr = re.sub(rf"\bPart\s+{re.escape(str(part_id))}\b", object_token, raw_qstr)
    return raw_qstr


def _format_question_params(
    question_yaml: dict[str, Any],
    question_json: dict[str, Any],
    frame_idxs: list[int],
    referred_obj_ids: dict[int, set[str]],
) -> dict[str, str]:
    question_category = question_json["question_category"]
    question_params = question_yaml.get("question_params", {})
    part_replacements: list[tuple[str, str]] = []

    for key, value in question_params.items():
        values = _as_list(value)
        if len(values) != 1:
            raise ValueError(f"Unexpected value length for question param {key}")

        obj_id = str(values[0])
        suffix = "A" if question_category == "tracking" else ""
        question_params[key] = f"<object{obj_id}{suffix}>"
        referred_obj_ids.setdefault(frame_idxs[0], set()).add(obj_id)
        part_replacements.append((obj_id, question_params[key]))

    if QUESTION_TEMPLATE_REGISTRY is not None:
        template_type = question_json["template_type"]
        template_idx = question_json.get("template_idx", 0)
        templates = QUESTION_TEMPLATE_REGISTRY[question_category][template_type]
        selected_template = templates[template_idx] if isinstance(templates, list) else templates
        question_json["question"]["raw_qstr"] = selected_template.format(**question_params)
    else:
        raw_qstr = question_json["question"].get("raw_qstr", "")
        question_json["question"]["raw_qstr"] = _replace_part_mentions(raw_qstr, part_replacements)

    return question_params


def _get_option_type(options: dict[str, dict[str, Any]]) -> str:
    option_type = "str"
    first_raw = options["0"]["raw"]
    if isinstance(first_raw, list):
        if first_raw and isinstance(first_raw[0], list):
            option_type = "list[list]"
        else:
            option_type = "list"
    elif isinstance(first_raw, str) and first_raw.isdigit():
        option_type = "str(int)"
    return option_type


def _format_options(
    question_json: dict[str, Any],
    frame_idxs: list[int],
    referred_obj_ids: dict[int, set[str]],
) -> None:
    options = question_json["question"]["options"]
    option_type = _get_option_type(options)
    question_category = question_json["question_category"]

    for opt_idx in options:
        raw_option = options[opt_idx]["raw"]

        if raw_option in ["none", "yes", "no"]:
            continue
        if option_type == "str(int)" and not isinstance(raw_option, str):
            continue
        if option_type == "list" and any(isinstance(x, list) for x in raw_option):
            continue
        if option_type == "list[list]" and any(
            isinstance(x, list) for item in raw_option for x in item
        ):
            continue

        if option_type == "str(int)":
            options[opt_idx]["text"] = f"<object{raw_option}>"
            if question_category == "tracking":
                options[opt_idx]["text"] = f"<object{raw_option}B>"
                referred_obj_ids.setdefault(frame_idxs[1], set()).add(str(raw_option))
            else:
                referred_obj_ids.setdefault(frame_idxs[0], set()).add(str(raw_option))

        elif option_type == "list":
            options[opt_idx]["text"] = ", ".join(f"<object{obj_id}>" for obj_id in raw_option)
            if question_category == "tracking":
                raise ValueError("list type options are not supported for tracking questions")
            referred_obj_ids.setdefault(frame_idxs[0], set()).update(str(x) for x in raw_option)

        elif option_type == "list[list]":
            list_of_lists = []
            for obj_list in raw_option:
                if len(obj_list) != 2:
                    raise ValueError(
                        "Expected object list of length 2 for tracking question, "
                        f"got {len(obj_list)}"
                    )

                obj_strs = []
                for i, obj_id in enumerate(obj_list):
                    suffix = (
                        ("A" if i == 0 else "B") if question_category == "tracking" else ""
                    )
                    obj_strs.append(f"<object{obj_id}{suffix}>")
                if question_category == "tracking":
                    list_of_lists.append(f"{obj_strs[0]} matches {obj_strs[1]}")
                    referred_obj_ids.setdefault(frame_idxs[0], set()).add(str(obj_list[0]))
                    referred_obj_ids.setdefault(frame_idxs[1], set()).add(str(obj_list[1]))
                else:
                    list_of_lists.append(" and ".join(obj_strs))
                    referred_obj_ids.setdefault(frame_idxs[0], set()).update(
                        str(x) for x in obj_list
                    )

            options[opt_idx]["text"] = ", ".join(list_of_lists)

        options[opt_idx]["full_text"] = (
            options[opt_idx]["label"] + ". " + options[opt_idx]["text"]
        )
        if int(opt_idx) == question_json["question"]["correct_option"]["idx"]:
            question_json["question"]["correct_option"]["text"] = options[opt_idx]["text"]
            question_json["question"]["correct_option"]["full_text"] = options[opt_idx]["full_text"]

    question_json["question"]["options"] = options
    question_json["question"]["qstr"] = (
        question_json["question"]["raw_qstr"]
        + "\nOptions:\n"
        + "\n".join(options[opt]["full_text"] for opt in options)
    )


def _load_prompt_masks(
    question_json: dict[str, Any],
    prompt_mask_dir: str,
    frame_idxs: list[int],
) -> dict[str, dict[str, Any]]:
    video_id = question_json["video_id"]
    category = question_json.get("vid_category", "unknown")
    name = question_json.get("furniture_name", "unknown")
    mask_fn = osp.join(prompt_mask_dir, category, name, video_id, f"{video_id}.json")

    with open(mask_fn, "r") as m_f:
        mask_json = json.load(m_f)["manual"]

    return {str(idx): mask_json.get(str(idx), {}) for idx in frame_idxs}


def _append_nontracking_prompt_masks(
    prompt_mask_list_for_question: list[dict[str, Any]],
    prepend_parts: list[str],
    prompt_masks: dict[str, dict[str, Any]],
    frame_idx: int,
    part_id_list: list[str],
    files_with_visibility_issues: list[str],
    question_yaml_fn: str,
) -> None:
    for part_id in part_id_list:
        try:
            mask_rle = prompt_masks[str(frame_idx)][str(part_id)]
        except KeyError:
            files_with_visibility_issues.append(question_yaml_fn)
            mask_rle = _zero_mask_like(prompt_masks[str(frame_idx)])

        prompt_mask_list_for_question.append(
            {
                "frame_idx": frame_idx,
                "part_id": part_id,
                "mask_rle": mask_rle,
            }
        )
        prepend_parts.append(f"<object{part_id}><region>")


def _append_tracking_prompt_masks(
    prompt_mask_list_for_question: list[dict[str, Any]],
    prepend_parts: list[str],
    prompt_masks: dict[str, dict[str, Any]],
    frame_idx: int,
    frame_position: int,
    part_id_list: list[str],
    jumble_map: dict[str, str],
    files_with_visibility_issues: list[str],
    question_yaml_fn: str,
) -> None:
    for part_id in part_id_list:
        if frame_position == 0:
            mapped_part_id = part_id
        else:
            mapped_part_id = next(
                (k for k, v in jumble_map.items() if v == str(part_id)),
                part_id,
            )

        try:
            mask_rle = prompt_masks[str(frame_idx)][str(mapped_part_id)]
        except KeyError:
            files_with_visibility_issues.append(question_yaml_fn)
            mask_rle = _zero_mask_like(prompt_masks[str(frame_idx)])

        prompt_mask_list_for_question.append(
            {
                "frame_idx": frame_idx,
                "part_id": part_id,
                "mask_rle": mask_rle,
            }
        )
        prepend_parts.append(f"<object{part_id}{'A' if frame_position == 0 else 'B'}><region>")


def _write_debug_prompt_images(
    question_json: dict[str, Any],
    question_yaml_fn: str,
    output_jsonl: str,
    prompt_mask_list_for_question: list[dict[str, Any]],
    rgb_frame_dir: str,
    frame_idxs: list[int],
    line_idx: int,
) -> None:
    from src.eval.prompts.media.components.som_visualization import generate_som_prompt_image

    video_id = question_json["video_id"]
    category = question_json.get("vid_category", "unknown")
    name = question_json.get("furniture_name", "unknown")

    for frame_idx in frame_idxs:
        masks_for_idx = [
            pm for pm in prompt_mask_list_for_question if pm["frame_idx"] == frame_idx
        ]
        mask_dict = {pm["part_id"]: pm["mask_rle"] for pm in masks_for_idx}
        rgb_frame_fn = osp.join(rgb_frame_dir, category, name, video_id, f"{frame_idx}.jpg")
        rgb = iio.imread(rgb_frame_fn)
        overlay_img = generate_som_prompt_image(
            img=rgb,
            masks=mask_dict,
            high_contrast_colors=True,
        )
        if isinstance(overlay_img, tuple):
            overlay_img = overlay_img[0]

        output_dir = osp.dirname(output_jsonl) or "."
        question_id = osp.basename(question_yaml_fn).split(".")[0]
        debug_img_name = f"{question_id}_prompt_{frame_idx:03d}_{line_idx:04d}.jpg"
        prompt_img_save_fn = osp.join(
            output_dir,
            "prompt_images2",
            debug_img_name,
        )
        os.makedirs(osp.dirname(prompt_img_save_fn), exist_ok=True)
        iio.imwrite(prompt_img_save_fn, np.array(overlay_img))


def convert_annotations(
    input_jsonl: str,
    output_jsonl: str,
    question_dir: str,
    prompt_mask_dir: str,
    debug: bool = False,
    rgb_frame_dir: str | None = None,
    debug_line_idx: int | None = 218,
    overwrite: bool = True,
) -> None:
    files_with_visibility_issues: list[str] = []

    with open(input_jsonl, "r") as f:
        lines = f.readlines()

    question_yaml_files = _sorted_question_yaml_files(question_dir)
    if len(question_yaml_files) != len(lines):
        raise ValueError(
            f"Mismatch between question JSONL rows ({len(lines)}) and YAML files "
            f"({len(question_yaml_files)})"
        )

    output_dir = osp.dirname(output_jsonl)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if overwrite and osp.exists(output_jsonl):
        os.remove(output_jsonl)

    for line_idx, line in tqdm(enumerate(lines), total=len(lines)):
        question_json = json.loads(line)
        question_yaml_fn = question_yaml_files[line_idx]
        with open(question_yaml_fn, "r") as q_f:
            question_yaml = yaml.safe_load(q_f)

        frame_idxs = question_yaml.get("frame_idx", [])
        if isinstance(frame_idxs, int):
            frame_idxs = [frame_idxs]

        referred_obj_ids: dict[int, set[str]] = {}
        _format_question_params(
            question_yaml=question_yaml,
            question_json=question_json,
            frame_idxs=frame_idxs,
            referred_obj_ids=referred_obj_ids,
        )
        prompt_masks = _load_prompt_masks(question_json, prompt_mask_dir, frame_idxs)
        if len(prompt_masks) != len(frame_idxs):
            raise ValueError(f"Mismatch in frame indices for video {question_json['video_id']}")

        _format_options(question_json, frame_idxs, referred_obj_ids)

        prepend_parts: list[str] = []
        prompt_mask_list_for_question: list[dict[str, Any]] = []
        question_category = question_json["question_category"]
        jumble_map = question_yaml.get("jumble_map", {})
        if isinstance(jumble_map, list):
            jumble_map = {
                k: v for map_entry in jumble_map for k, v in map_entry.items()
            }

        for frame_position, frame_idx in enumerate(frame_idxs):
            part_id_set = set(prompt_masks[str(frame_idx)].keys())
            part_id_list = sorted(part_id_set)

            if question_category == "tracking":
                referred_for_frame = referred_obj_ids.get(frame_idx, set())
                if frame_position == 1:
                    part_id_list = sorted(referred_for_frame)
                elif frame_position == 0 and referred_for_frame - part_id_set:
                    files_with_visibility_issues.append(question_yaml_fn)

                _append_tracking_prompt_masks(
                    prompt_mask_list_for_question=prompt_mask_list_for_question,
                    prepend_parts=prepend_parts,
                    prompt_masks=prompt_masks,
                    frame_idx=frame_idx,
                    frame_position=frame_position,
                    part_id_list=part_id_list,
                    jumble_map=jumble_map,
                    files_with_visibility_issues=files_with_visibility_issues,
                    question_yaml_fn=question_yaml_fn,
                )
            else:
                _append_nontracking_prompt_masks(
                    prompt_mask_list_for_question=prompt_mask_list_for_question,
                    prepend_parts=prepend_parts,
                    prompt_masks=prompt_masks,
                    frame_idx=frame_idx,
                    part_id_list=part_id_list,
                    files_with_visibility_issues=files_with_visibility_issues,
                    question_yaml_fn=question_yaml_fn,
                )

        prepend_str = (
            "<video>\nThis question is about the following objects in the video: "
            + ", ".join(prepend_parts)
            + ".\n"
        )
        append_str = "Answer with the option's letter from the given choices directly."
        question_json["question"]["qstr"] = (
            prepend_str + question_json["question"]["qstr"] + "\n" + append_str
        )
        question_json["question"]["prompt_masks"] = prompt_mask_list_for_question

        should_debug = debug and (debug_line_idx is None or line_idx == debug_line_idx)
        if should_debug:
            if rgb_frame_dir is None:
                raise ValueError("rgb_frame_dir must be provided when debug=True")
            _write_debug_prompt_images(
                question_json=question_json,
                question_yaml_fn=question_yaml_fn,
                output_jsonl=output_jsonl,
                prompt_mask_list_for_question=prompt_mask_list_for_question,
                rgb_frame_dir=rgb_frame_dir,
                frame_idxs=frame_idxs,
                line_idx=line_idx,
            )

        with open(output_jsonl, "a") as out_f:
            out_f.write(json.dumps(question_json) + "\n")

        if should_debug:
            print(json.dumps(question_json, indent=4))
            break

    debug_fn = output_jsonl.replace(".jsonl", "_debug.json")
    with open(debug_fn, "w") as debug_f:
        debug_f.write(json.dumps(files_with_visibility_issues, indent=4) + "\n")


def parse_args() -> argparse.Namespace:
    data_dir = ROOT / "data"
    parser = argparse.ArgumentParser(
        description="Convert Flat-Pack Bench questions to the VideoRefer prompt-mask format."
    )
    parser.add_argument(
        "--input-jsonl",
        default=str(data_dir / "questions" / "questions.jsonl"),
        help="Input Flat-Pack Bench questions JSONL.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=str(data_dir / "questions" / "questions_videorefer_format.jsonl"),
        help="Output VideoRefer-format JSONL.",
    )
    parser.add_argument(
        "--question-dir",
        default=str(data_dir / "questions" / "yamls"),
        help="Directory containing source YAML question files.",
    )
    parser.add_argument(
        "--prompt-mask-dir",
        default=str(data_dir / "segmentation-masks"),
        help="Directory containing prompt mask JSON files.",
    )
    parser.add_argument(
        "--rgb-frame-dir",
        help="Directory containing RGB frames, used only with --debug.",
    )
    parser.add_argument("--debug", action="store_true", help="Write debug prompt images.")
    parser.add_argument(
        "--debug-line-idx",
        type=int,
        default=218,
        help="JSONL row to render when --debug is set. Use -1 to render all rows.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to an existing output file instead of overwriting it.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_annotations(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        question_dir=args.question_dir,
        prompt_mask_dir=args.prompt_mask_dir,
        rgb_frame_dir=args.rgb_frame_dir,
        debug=args.debug,
        debug_line_idx=None if args.debug_line_idx < 0 else args.debug_line_idx,
        overwrite=not args.append,
    )
