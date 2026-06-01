import pyrootutils

root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from typing import Any, Dict, Optional
import argparse
import json
import os
import os.path as osp
import re

import numpy as np
import torch
from pycocotools.mask import decode
from tqdm import tqdm

from src.eval.models.base_model import BaseModel


DEFAULT_MODEL_PATH = "DAMO-NLP-SG/VideoRefer-VideoLLaMA3-7B"


def post_process_response(response: Any) -> str:
    if response is None:
        return ""
    if not isinstance(response, str):
        response = str(response)

    stripped_response = response.strip()
    match = re.search(r"\b([A-Z])\b", stripped_response)
    if match:
        return match.group(1)
    return stripped_response


class VideoRefer(BaseModel):
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        max_frames: int = 768,
        fps: float = 1,
        load_at_init: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.max_frames = max_frames
        self.fps = fps

        self.model = None
        self.processor = None
        self.tokenizer = None
        self._load_video = None
        self._get_model_output = None

        if load_at_init:
            self.setup_model()

    def setup_model(self) -> None:
        if self.model is not None:
            return

        from videorefer_videollama3 import (
            disable_torch_init,
            get_model_output,
            model_init,
        )
        from videorefer_videollama3.mm_utils import load_video

        disable_torch_init()
        self.model, self.processor, self.tokenizer = model_init(self.model_path)
        self._load_video = load_video
        self._get_model_output = get_model_output

    @staticmethod
    def _resolve_video_path(sample: Dict[str, Any], video_dir: Optional[str]) -> str:
        if video_dir is None:
            media_dir = sample.get("media_dir")
            video = sample.get("video")
            if not media_dir or not video:
                raise ValueError(
                    "video_dir is required unless each sample has media_dir and video fields."
                )
            return osp.join(media_dir, video)

        category = sample["vid_category"]
        name = sample["furniture_name"]
        video_id = sample["video_id"]
        return osp.join(video_dir, category, name, video_id, f"{video_id}.mp4")

    @staticmethod
    def _frame_idxs(sample: Dict[str, Any]) -> list[int]:
        frame_idxs = sample["frame_idx"]
        if isinstance(frame_idxs, int):
            frame_idxs = [frame_idxs]
        return frame_idxs

    @staticmethod
    def _prompt_masks_to_tensor(
        sample: Dict[str, Any],
        frame_idxs: list[int],
    ) -> tuple[torch.Tensor, list[int]]:
        prompt_masks = sample.get("question", {}).get("prompt_masks")
        if not prompt_masks:
            raise ValueError(
                "Sample does not contain question.prompt_masks. Run "
                "convert_data_format.convert_annotations before VideoRefer inference."
            )

        frame_idxs_for_masks = [item["frame_idx"] for item in prompt_masks]
        mask_ids = [frame_idxs.index(frame_idx) for frame_idx in frame_idxs_for_masks]
        masks = np.array([decode(item["mask_rle"]) for item in prompt_masks])
        return torch.from_numpy(masks).to(torch.uint8), mask_ids

    def infer_sample(
        self,
        sample: Dict[str, Any],
        video_dir: Optional[str] = None,
        max_frames: Optional[int] = None,
        fps: Optional[float] = None,
    ) -> str:
        self.setup_model()

        frame_idxs = self._frame_idxs(sample)
        masks, mask_ids = self._prompt_masks_to_tensor(sample, frame_idxs)
        video_path = self._resolve_video_path(sample, video_dir)

        video_tensor = self._load_video(
            video_path,
            fps=self.fps if fps is None else fps,
            max_frames=self.max_frames if max_frames is None else max_frames,
            frame_ids=frame_idxs,
        )

        return self._get_model_output(
            video_tensor,
            sample["question"]["qstr"],
            model=self.model,
            tokenizer=self.tokenizer,
            masks=masks,
            mask_ids=mask_ids,
            modal="video",
        )

    def forward(
        self,
        sample: Dict[str, Any],
        video_dir: Optional[str] = None,
        max_frames: Optional[int] = None,
        fps: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        if isinstance(sample, list):
            raise ValueError(
                "VideoRefer expects a converted question sample, not a rendered "
                "conversation. Use run_inference_with_conversion for this model."
            )
        return self.infer_sample(
            sample=sample,
            video_dir=video_dir,
            max_frames=max_frames,
            fps=fps,
        )

    def post_process_response(self, response: Any) -> str:
        return post_process_response(response)


def _default_converted_dataset_fn(dataset_fn: str) -> str:
    base, ext = osp.splitext(dataset_fn)
    if ext != ".jsonl":
        return f"{dataset_fn}_videorefer_format.jsonl"
    return f"{base}_videorefer_format.jsonl"


def run_inference_on_dataset(
    dataset_fn: str,
    output_fn: str,
    video_dir: Optional[str],
    model: Optional[VideoRefer] = None,
    model_path: str = DEFAULT_MODEL_PATH,
    max_frames: int = 768,
    fps: float = 1,
    limit: Optional[int] = None,
    append_output: bool = False,
) -> None:
    model = model or VideoRefer(
        model_path=model_path,
        max_frames=max_frames,
        fps=fps,
    )

    output_dir = osp.dirname(output_fn)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if not append_output and osp.exists(output_fn):
        os.remove(output_fn)

    with open(dataset_fn, "r") as f:
        pbar = tqdm(enumerate(f), desc="Running VideoRefer inference")
        for line_idx, line in pbar:
            if limit is not None and line_idx >= limit:
                break

            sample = json.loads(line.strip())
            with torch.no_grad():
                raw_response = model.infer_sample(
                    sample=sample,
                    video_dir=video_dir,
                    max_frames=max_frames,
                    fps=fps,
                )
            response = model.post_process_response(raw_response)

            result = {
                "conv_id": sample["qid"],
                "question": sample,
                "response": response,
                "raw_response": raw_response,
                "post_processed_response": response,
                "qid": sample["qid"],
                "qid_flat": sample["qid_flat"],
            }
            with open(output_fn, "a") as out_f:
                out_f.write(json.dumps(result) + "\n")


def run_inference_with_conversion(
    dataset_fn: str,
    output_fn: str,
    video_dir: Optional[str],
    converted_dataset_fn: Optional[str] = None,
    question_dir: Optional[str] = None,
    prompt_mask_dir: Optional[str] = None,
    rgb_frame_dir: Optional[str] = None,
    skip_convert: bool = False,
    convert_debug: bool = False,
    convert_debug_line_idx: Optional[int] = 218,
    append_converted: bool = False,
    **inference_kwargs: Any,
) -> str:
    dataset_for_inference = (
        converted_dataset_fn if skip_convert and converted_dataset_fn else dataset_fn
    )

    if not skip_convert:
        from src.eval.models.model_utils.videorefer.convert_data_format import (
            convert_annotations,
        )

        data_dir = root / "data"
        converted_dataset_fn = converted_dataset_fn or _default_converted_dataset_fn(dataset_fn)
        question_dir = question_dir or str(data_dir / "questions" / "yamls")
        prompt_mask_dir = prompt_mask_dir or str(data_dir / "segmentation-masks")
        rgb_frame_dir = rgb_frame_dir or str(data_dir / "rgb-frames")

        convert_annotations(
            input_jsonl=dataset_fn,
            output_jsonl=converted_dataset_fn,
            question_dir=question_dir,
            prompt_mask_dir=prompt_mask_dir,
            debug=convert_debug,
            rgb_frame_dir=rgb_frame_dir,
            debug_line_idx=convert_debug_line_idx,
            overwrite=not append_converted,
        )
        dataset_for_inference = converted_dataset_fn

    run_inference_on_dataset(
        dataset_fn=dataset_for_inference,
        output_fn=output_fn,
        video_dir=video_dir,
        **inference_kwargs,
    )
    return dataset_for_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Flat-Pack Bench data and run VideoRefer inference."
    )
    parser.add_argument(
        "--dataset-fn",
        default=str(root / "data" / "questions" / "questions.jsonl"),
        help="Input Flat-Pack Bench questions JSONL.",
    )
    parser.add_argument(
        "--converted-dataset-fn",
        default=None,
        help="Where to write/read the VideoRefer-format JSONL.",
    )
    parser.add_argument(
        "--output-fn",
        default="./output.jsonl",
        help="Output responses JSONL.",
    )
    parser.add_argument(
        "--video-dir",
        default=str(root / "data" / "videos" / "keyframe-video" / "fps-1"),
        help="Directory containing videos as category/furniture_name/video_id/video_id.mp4.",
    )
    parser.add_argument(
        "--question-dir",
        default=str(root / "data" / "questions" / "yamls"),
        help="Directory containing source question YAMLs.",
    )
    parser.add_argument(
        "--prompt-mask-dir",
        default=str(root / "data" / "segmentation-masks"),
        help="Directory containing prompt mask JSON files.",
    )
    parser.add_argument(
        "--rgb-frame-dir",
        default=str(root / "data" / "rgb-frames"),
        help="RGB frame directory used only for converter debug images.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="VideoRefer model path.",
    )
    parser.add_argument("--max-frames", type=int, default=768)
    parser.add_argument("--fps", type=float, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of examples to run.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run only the first three examples unless --limit is also set.",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip conversion and assume --dataset-fn already has prompt masks.",
    )
    parser.add_argument(
        "--convert-debug",
        action="store_true",
        help="Render converter debug prompt images.",
    )
    parser.add_argument(
        "--convert-debug-line-idx",
        type=int,
        default=218,
        help="Converter debug row. Use -1 to debug all rows.",
    )
    parser.add_argument(
        "--append-output",
        action="store_true",
        help="Append responses instead of overwriting the output file.",
    )
    parser.add_argument(
        "--append-converted",
        action="store_true",
        help="Append converted rows instead of overwriting the converted JSONL.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    limit = args.limit
    if args.debug and limit is None:
        limit = 3

    run_inference_with_conversion(
        dataset_fn=args.dataset_fn,
        output_fn=args.output_fn,
        video_dir=args.video_dir,
        converted_dataset_fn=args.converted_dataset_fn,
        question_dir=args.question_dir,
        prompt_mask_dir=args.prompt_mask_dir,
        rgb_frame_dir=args.rgb_frame_dir,
        skip_convert=args.skip_convert,
        convert_debug=args.convert_debug,
        convert_debug_line_idx=None
        if args.convert_debug_line_idx < 0
        else args.convert_debug_line_idx,
        append_converted=args.append_converted,
        model_path=args.model_path,
        max_frames=args.max_frames,
        fps=args.fps,
        limit=limit,
        append_output=args.append_output,
    )
