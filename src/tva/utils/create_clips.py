import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    pythonpath=True,
    dotenv=True,
)

import argparse
import glob
import json
import os
import os.path as osp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

from tqdm import tqdm

from src.tva.utils.video import split_video_by_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=str, required=True, help="Directory containing video files.")
    parser.add_argument(
        "--frames-metadata-dir",
        type=str,
        required=True,
        help="Directory containing frames metadata files.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save trimmed video files.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers. Defaults to the number of CPU cores.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=256,
        help="Maximum number of frames per split video segment.",
    )
    return parser.parse_args()


def load_video_metadata_pairs(video_dir: str, metadata_dir: str) -> List[Tuple[str, str]]:
    video_fns = glob.glob(osp.join(video_dir, "*", "*", "*", "*.mp4"))
    video_ids = [osp.basename(x).split(".")[0] for x in video_fns]
    metadata_fns = glob.glob(osp.join(metadata_dir, "*", "*", "*", "*_frames_metadata.jsonl"))
    metadata_fns = [i for i in metadata_fns if osp.basename(i).split("_frames_metadata.jsonl")[0] in video_ids]

    video_fns = sorted(video_fns, key=lambda x: osp.basename(x).split(".")[0])
    metadata_fns = sorted(metadata_fns, key=lambda x: osp.basename(x).split("_frames_metadata.jsonl")[0])

    assert len(video_fns) == len(metadata_fns), (
        f"Number of video files ({len(video_fns)}) and metadata files ({len(metadata_fns)}) do not match."
    )
    assert all(
        osp.basename(i).split("_frames_metadata.jsonl")[0] == osp.basename(j).split(".")[0]
        for i, j in zip(metadata_fns, video_fns)
    ), "Video files and metadata files do not match."

    return list(zip(video_fns, metadata_fns))


def _read_frame_time_bounds(metadata_fn: str) -> Tuple[float, float]:
    min_frame_time = None
    max_frame_time = None
    with open(metadata_fn, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            frame_time = json.loads(line)["frame_time"]
            if min_frame_time is None:
                min_frame_time = frame_time
            max_frame_time = frame_time

    if min_frame_time is None or max_frame_time is None:
        raise ValueError(f"No frame_time entries found in metadata file {metadata_fn}.")

    return float(min_frame_time), float(max_frame_time)


def process_video(video_fn: str, metadata_fn: str, video_dir: str, output_dir: str, max_frames: int=256) -> str:
    if not osp.exists(metadata_fn):
        raise ValueError(f"Metadata file {metadata_fn} does not exist.")

    out_dir = osp.join(output_dir, osp.relpath(osp.dirname(video_fn), video_dir))
    os.makedirs(out_dir, exist_ok=True)

    min_frame_time, max_frame_time = _read_frame_time_bounds(metadata_fn)

    split_video_by_frames(
        input_path=video_fn,
        out_dir=out_dir,
        max_frames=max_frames,
        crf=18,
        overlap=1,
    )
    return video_fn


def run_parallel(
    jobs: List[Tuple[str, str]],
    video_dir: str,
    output_dir: str,
    num_workers: int,
    max_frames: int = 256,
) -> None:
    if num_workers == 1 or len(jobs) == 1:
        for idx, (video_fn, metadata_fn) in enumerate(tqdm(jobs, desc="Processing videos", unit="video"), start=1):
            process_video(video_fn, metadata_fn, video_dir, output_dir, max_frames)
        return

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_video, video_fn, metadata_fn, video_dir, output_dir, max_frames): video_fn
            for video_fn, metadata_fn in jobs
        }

        with tqdm(total=len(futures), desc="Processing videos", unit="video") as pbar:
            for future in as_completed(futures):
                video_fn = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    raise RuntimeError(f"Failed to process video {video_fn}") from exc
                pbar.set_postfix_str(osp.basename(video_fn))
                pbar.update(1)


def main() -> None:
    args = parse_args()

    if args.num_workers is not None and args.num_workers <= 0:
        raise ValueError("--num-workers must be a positive integer or omitted.")

    os.makedirs(args.output_dir, exist_ok=True)

    jobs = load_video_metadata_pairs(args.video_dir, args.frames_metadata_dir)

    max_workers = args.num_workers or os.cpu_count() or 1
    max_workers = min(max_workers, len(jobs)) or 1

    run_parallel(jobs, args.video_dir, args.output_dir, max_workers, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
