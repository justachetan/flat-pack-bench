import os
import os.path as osp
import math
import shutil
import yaml
import json
from bisect import bisect_left
from typing import Iterable, Optional, Sequence, Set, Union, Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import av
import glob

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_video_handle(
    vid_fn: str
):
    """
    Load a video file and return the container handle.

    Args:
        vid_fn (str): Path to the video file.
    Returns:
        cv2.VideoCapture: Video capture handle.
    """
    cap = cv2.VideoCapture(vid_fn)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {vid_fn}")
    return cap

def subspl_video_streaming(
    video_path: str,
    keep_every: int,
    keep_times_sec: Optional[Set[float]] = None,
    keyframe_idxs_for_keep_times: Optional[Set[int]] = None,
    trim_video: bool = False,
    trim_delta_sec: float = 1,
    clip_ends: bool = False,
    keyframe_timestamps: Optional[Sequence[float]] = None,
) -> Iterable[Tuple[int, int, Optional[int], float, "np.ndarray"]]:
    """
    Subsample video frames from a video file.

    Args:
        video_path (str): Path to the video file.
        keep_every (int): Subsampling factor, keep every N-th frame.
        keep_times_sec (Optional[Set[float]], optional): Specific timestamps to always keep. Defaults to None.
        trim_video (bool, optional): Whether to trim the video based on metadata timestamps. Defaults to False.
        trim_delta_sec (float, optional): Delta in seconds to adjust trimming. Defaults to 1.
        clip_ends (bool, optional): Whether to clip the first and last frames. Defaults to False.
        keyframe_timestamps (Optional[Sequence[float]], optional): List of keyframe timestamps to decide gaps. Defaults to None.
    Yields:
        Iterable[Tuple[int, any]]: Yields tuples of (frame_index, frame).
    """

    keep_ts_ptr = 0
    vid_cap = load_video_handle(video_path)
    keep_times_sec = sorted(list(keep_times_sec)) if keep_times_sec is not None else None
    keyframe_idxs_for_keep_times = sorted(list(keyframe_idxs_for_keep_times)) \
        if keyframe_idxs_for_keep_times is not None else None
    
    # print(keep_times_sec)
    # with open(metadata_path, 'r') as f:
    #     meta_ts_s = [json.loads(line.strip())["frame_time"] for line in f]
    meta_ts_s = keyframe_timestamps if keyframe_timestamps is not None else []
    # import ipdb; ipdb.set_trace()
    meta_ts_s = np.array(sorted(meta_ts_s))
    meta_ts_ptr = 0

    if keyframe_idxs_for_keep_times is None:
        keyframe_idxs_for_keep_times = [i for i in range(len(meta_ts_s)) if meta_ts_s[i] in keep_times_sec] \
            if keep_times_sec is not None else []
    

    gaps_start_sec = None
    gaps_end_sec = None
    if trim_video and len(meta_ts_s) > 1:
        jumps = meta_ts_s[1:] - meta_ts_s[:-1]
        gap_indices = np.where(jumps > trim_delta_sec)[0]
        if len(gap_indices) > 0:
            gaps_end_sec = meta_ts_s[gap_indices + 1]
            gaps_start_sec = meta_ts_s[gap_indices]

    i = 0
    subspl_idx = 0
    last_ts_ms = -1
    
    while True:

        cur_prompt_frame_idx = None # if the frame is a prompt frame, this is its index in the keyframe list
        cur_prompt_frame_idx_in_keyframes = None
        cur_prompt_frame_ts_in_keyframes = None

        grabbed = vid_cap.grab()
        if not grabbed:
            break

        # Timestamp (milliseconds) for the frame that has just been grabbed.
        ts_ms = vid_cap.get(cv2.CAP_PROP_POS_MSEC)
        ts_s = ts_ms / 1000.0
        ok, frame = vid_cap.retrieve()
        if not ok:
            # corrupted frame, continue
            continue

        keep_frame = True # meant to skip gaps when trimming
        is_prompt_frame = keep_times_sec[keep_ts_ptr] <= ts_s \
            if keep_times_sec is not None and keep_ts_ptr < len(keep_times_sec) else False
        
        # if current frame is a prompt frame, always keep it
        if is_prompt_frame:
            keep_frame = True
            cur_prompt_frame_idx = keep_ts_ptr
            cur_prompt_frame_idx_in_keyframes = keyframe_idxs_for_keep_times[keep_ts_ptr]
            cur_prompt_frame_ts_in_keyframes = keep_times_sec[keep_ts_ptr]
            # yield (i, keep_ts_ptr, ts_s,  frame)
            keep_ts_ptr += 1

        elif trim_video:
            if clip_ends:
                if ts_s < meta_ts_s[0] or ts_s > meta_ts_s[-1]:
                    keep_frame = False
            if gaps_start_sec is not None and gaps_end_sec is not None:
                if np.any((ts_s > gaps_start_sec) & (ts_s < gaps_end_sec)):
                    keep_frame = False

        if is_prompt_frame or ((i % keep_every) == 0 and keep_frame):
            yield (i, subspl_idx, cur_prompt_frame_idx, cur_prompt_frame_idx_in_keyframes,\
                    ts_s, cur_prompt_frame_ts_in_keyframes, frame)
            subspl_idx += 1
        i += 1
        last_ts_ms = ts_ms
    # print(keep_ts_ptr, len(keep_times_sec))
    vid_cap.release()


def _process_single_video(
    video_path: str,
    rel_path: str,
    meta_path: str,
    out_dir_root: str,
    keep_every: int,
    trim_video: bool,
    trim_delta_sec: float,
    clip_ends: bool,
    overwrite: bool,
    dryrun: bool,
    preserve_indices: Set[int],
    max_frames_in_clip: int = 256,
    overlap: int = 1,
    debug: bool = False,
) -> None:
    video_id = osp.basename(video_path)[:-4]
    output_check_path = osp.dirname(osp.join(out_dir_root, rel_path))

    if osp.exists(output_check_path) and not overwrite:
        print(f"[SKIP] Output exists: {output_check_path}")
        return

    os.makedirs(osp.dirname(output_check_path), exist_ok=True)

    if not osp.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    keyframe_timestamps: List[float] = []
    keep_timestamps_sec: Optional[List[float]] = None
    with open(meta_path, "r") as f:
        for line_idx, line in enumerate(f):
            meta = json.loads(line.strip())
            if line_idx in preserve_indices:
                if keep_timestamps_sec is None:
                    keep_timestamps_sec = []
                keep_timestamps_sec.append(meta["frame_time"])
            keyframe_timestamps.append(meta["frame_time"])

    if keep_timestamps_sec is not None:
        print(f"  Loaded {len(keep_timestamps_sec)} timestamps to preserve from {meta_path}")

    print(f"Processing:\n  IN:  {video_path}\n  OUT: {output_check_path}")

    if dryrun:
        return

    frames_out_path = osp.join(output_check_path, "frames")
    clips_out_path = osp.join(output_check_path, "clips")
    os.makedirs(frames_out_path, exist_ok=True)
    os.makedirs(clips_out_path, exist_ok=True)

    frame_iter = subspl_video_streaming(
        video_path=video_path,
        keep_every=keep_every,
        keep_times_sec=keep_timestamps_sec if keep_timestamps_sec is not None else None,
        keyframe_idxs_for_keep_times=preserve_indices,
        trim_video=trim_video,
        trim_delta_sec=trim_delta_sec,
        clip_ends=clip_ends,
        keyframe_timestamps=keyframe_timestamps,
    )

    video_frames_metadata = []
    for i, subspl_idx, cur_prompt_frame_idx, cur_prompt_frame_idx_in_keyframes, ts_s, cur_prompt_frame_ts_in_keyframes, frame in frame_iter:
        video_frames_metadata.append({
            "original_frame_idx": i,
            "subsampled_frame_idx": subspl_idx,
            "timestamp_sec": ts_s,
            "is_prompt_frame": cur_prompt_frame_idx is not None,
            "prompt_frame_idx": cur_prompt_frame_idx,
            "keyframe_idx": cur_prompt_frame_idx_in_keyframes,
            "prompt_frame_timestamp_sec": cur_prompt_frame_ts_in_keyframes,
        })
        frame_fn = osp.join(frames_out_path, f"{subspl_idx:05d}.jpg")
        cv2.imwrite(frame_fn, frame)
        
        clip_idx = subspl_idx // max_frames_in_clip
        clip_dir = osp.join(clips_out_path, f"clip_{clip_idx:04d}")
        os.makedirs(clip_dir, exist_ok=True)
        clip_frame_idx = None

        if clip_idx != 0:
            for ov in range(overlap):
                prev_clip_idx = clip_idx - 1
                num_frames_in_prev_clip = len(
                    os.listdir(osp.join(clips_out_path, f"clip_{prev_clip_idx:04d}"))
                )
                prev_clip_frame_idx = num_frames_in_prev_clip - overlap + ov
                prev_clip_frame_fn = osp.join(
                    clips_out_path, f"clip_{prev_clip_idx:04d}", f"{prev_clip_frame_idx:03d}.jpg")
                if osp.exists(prev_clip_frame_fn):
                    shutil.copy(prev_clip_frame_fn,
                        osp.join(clip_dir, f"{ov:03d}.jpg"))
        
            clip_frame_idx = (subspl_idx % max_frames_in_clip) + overlap
        else:
            clip_frame_idx = subspl_idx % max_frames_in_clip
        clip_frame_fn = osp.join(clip_dir, f"{clip_frame_idx:03d}.jpg")
        cv2.imwrite(clip_frame_fn, frame)


    if not video_frames_metadata:
        print(f"[WARN] No frames were written for {video_path}")
        return

    meta_out_path = osp.join(osp.dirname(frames_out_path), "subspl_frames_metadata.csv")
    meta_df = pd.DataFrame(video_frames_metadata, columns=list(video_frames_metadata[0].keys()))
    # meta_df.fillna(value={"prompt_frame_idx": -1}, inplace=True)
    meta_df["prompt_frame_idx"] = meta_df["prompt_frame_idx"].fillna(-1).astype(int)
    meta_df["keyframe_idx"] = meta_df["keyframe_idx"].fillna(-1).astype(int)
    meta_df["prompt_frame_timestamp_sec"] = meta_df["prompt_frame_timestamp_sec"].fillna(-1).astype(float)
    meta_df.to_csv(meta_out_path, index=False)

def main(
    video_dir: str,
    metadata_dir: str,
    question_dir: str,
    out_dir_root: str,
    keep_every: int,
    question_files: Optional[Sequence[str]] = None,
    video_ids: Optional[Sequence[str]] = None,
    trim_video: bool = False,
    trim_delta_sec: float = 1,
    clip_ends: bool = True,
    overwrite: bool = False,
    dryrun: bool = False,
    debug: bool = False,
    max_frames_in_clip: int = 256,
    overlap: int = 1,
    num_workers: Optional[int] = None,
) -> Iterable[Tuple[int, int, Optional[int], float, "np.ndarray"]]:
    """
    Main function to subsample video frames.
    Args:
        video_dir (str): Path to the video file.
        metadata_dir (str): Path to the metadata file containing timestamps.
        question_dir (str): Path to the directory containing question files.
        question_files (Optional[Sequence[str]]): Specific question files to process. If None, process all YAMLs under the question_dir.
        video_ids (Optional[Sequence[str]]): Specific video IDs to process. If None, process all videos with questions.
        keep_every (int): Subsampling factor, keep every N-th frame.
        keep_times_sec (Optional[Set[float]], optional): Specific timestamps to always keep. Defaults to None.
        trim_video (bool, optional): Whether to trim the video based on metadata timestamps. Defaults to False.
        trim_delta_sec (float, optional): Delta in seconds to adjust trimming. Defaults to 1.
        clip_ends (bool, optional): Whether to clip the first and last frames. Defaults to False.
        overwrite (bool, optional): Whether to overwrite existing output files. Defaults to False.
        max_frames_in_clip (int, optional): Maximum number of frames in each clip. Defaults to 256.
        overlap (int, optional): Number of overlapping frames between clips. Defaults to 1.
        dryrun (bool, optional): If set, do not write any output files. Defaults to False.
        debug (bool, optional): If set, run in debug mode with fewer files. Defaults to False.
        num_workers (Optional[int], optional): Number of parallel workers. Defaults to os.cpu_count().
    Yields:
        Iterable[Tuple[int, any]]: Yields tuples of (frame_index, subspl_idx, cur_prompt_frame_idx, timestamp, frame).
            frame_index (int): Index of the frame in the video.
            subspl_idx (int): Index of the frame after subsampling.
            cur_prompt_frame_idx (int or None): Index of the current prompt frame if applicable.
            timestamp (float): Timestamp of the frame in seconds.
            frame (np.ndarray): The video frame.
    """

    # first we collect the frame indices in each video that we need to preserve
    video_to_preserve_indices = {}
    if question_files:
        question_paths = []
        for qpath in question_files:
            candidate_path = qpath if osp.isabs(qpath) else osp.join(question_dir, qpath)
            candidate_path = osp.normpath(candidate_path)
            if not osp.isfile(candidate_path):
                print(f"[WARN] Question file not found or not a file, skipping: {candidate_path}")
                continue
            question_paths.append(candidate_path)
        # deduplicate while preserving order
        question_paths = list(dict.fromkeys(question_paths))
        if debug:
            question_paths = question_paths[:10]
        print(f"Found {len(question_paths)} question files from provided list.")
    else:
        question_paths = sorted(glob.glob(osp.join(question_dir, "*.yaml")))
        if debug:
            question_paths = question_paths[:10]
        print(f"Found {len(question_paths)} question files under {question_dir}")

    if not question_paths:
        print("No question files to process.")
        return

    video_filter: Optional[Set[str]] = None
    if video_ids:
        filtered_ids = [vid.strip() for vid in video_ids if vid and vid.strip()]
        video_filter = set(filtered_ids)
        if not video_filter:
            print("[WARN] No valid video IDs provided after filtering; exiting.")
            return
        print(f"Filtering to {len(video_filter)} video IDs.")

    for qpath in question_paths:
        with open(qpath, "r") as f:
            qdata = yaml.safe_load(f)
        video_id = qdata["video_id"]
        if video_filter is not None and video_id not in video_filter:
            continue
        frame_indices = qdata["frame_idx"]
        if video_id not in video_to_preserve_indices:
            video_to_preserve_indices[video_id] = set()
        if isinstance(frame_indices, int):
            frame_indices = [frame_indices]
        video_to_preserve_indices[video_id].update(frame_indices)
    print(video_to_preserve_indices)
    if video_filter is not None:
        missing_ids = video_filter.difference(video_to_preserve_indices.keys())
        if missing_ids:
            print(f"[WARN] No questions matched the following video IDs: {', '.join(sorted(missing_ids))}")

    if not video_to_preserve_indices:
        print("No questions matched the provided filters.")
        return

    video_paths = glob.glob(osp.join(video_dir, "*", "*", "*", "*.mp4"))
    print(f"Found {len(video_paths)} videos under {video_dir}")

    tasks = []
    for in_path in video_paths:
        rel_path = osp.relpath(in_path, video_dir)
        meta_path = osp.join(metadata_dir, rel_path[:-4] + "_frames_metadata.jsonl")
        video_id = osp.basename(in_path)[:-4]
        if video_id not in video_to_preserve_indices:
            print(f"[SKIP] No questions for video {video_id}, skipping.")
            continue

        tasks.append((in_path, rel_path, meta_path, video_to_preserve_indices[video_id]))

    if not tasks:
        print("Nothing to process.")
        return

    worker_count = 1
    if num_workers is None:
        cpu_count = os.cpu_count()
        worker_count = cpu_count if cpu_count is not None and cpu_count > 0 else 1
    else:
        worker_count = max(1, num_workers)

    if worker_count == 1:
        for in_path, rel_path, meta_path, preserve_indices in tqdm(
            tasks, desc="Processing videos", unit="video"):

            _process_single_video(
                video_path=in_path,
                rel_path=rel_path,
                meta_path=meta_path,
                out_dir_root=out_dir_root,
                keep_every=keep_every,
                trim_video=trim_video,
                trim_delta_sec=trim_delta_sec,
                clip_ends=clip_ends,
                overwrite=overwrite,
                dryrun=dryrun,
                preserve_indices=preserve_indices,
                max_frames_in_clip=max_frames_in_clip,
                overlap=overlap,
                debug=debug,
            )
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    _process_single_video,
                    in_path,
                    rel_path,
                    meta_path,
                    out_dir_root,
                    keep_every,
                    trim_video,
                    trim_delta_sec,
                    clip_ends,
                    overwrite,
                    dryrun,
                    preserve_indices,
                    max_frames_in_clip,
                    overlap,
                    debug
                )
                for (in_path, rel_path, meta_path, preserve_indices) in tasks
            ]
            with tqdm(total=len(futures), desc="Processing videos", unit="video") as pbar:
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Subsample video frames with options to preserve specific frames.")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing input videos.")
    parser.add_argument("--metadata_dir", type=str, required=True, help="Directory containing video metadata files.")
    parser.add_argument("--question_dir", type=str, required=True, help="Directory containing question YAML files.")
    parser.add_argument(
        "--question_files",
        type=str,
        nargs="+",
        default=None,
        help="Specific question YAML files to process. Paths can be absolute or relative to --question_dir.",
    )
    parser.add_argument(
        "--video_ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific video IDs to process.",
    )
    parser.add_argument("--out_dir_root", type=str, required=True, help="Output directory root for subsampled frames.")
    parser.add_argument("--keep_every", type=int, default=4, help="Subsampling factor, keep every N-th frame.")
    parser.add_argument("--trim_video", action="store_true", help="Whether to trim the video based on metadata timestamps.")
    parser.add_argument("--trim_delta_sec", type=float, default=1.0, help="Delta in seconds to adjust trimming.")
    parser.add_argument("--clip_ends", action="store_true", help="Whether to clip the first and last frames.")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing output files.")
    parser.add_argument("--dryrun", action="store_true", help="If set, do not write any output files.")
    parser.add_argument("--debug", action="store_true", help="If set, run in debug mode with fewer files.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of parallel workers. Defaults to os.cpu_count().")
    parser.add_argument("--max_frames_in_clip", type=int, default=256, help="Maximum number of frames in each clip.")
    parser.add_argument("--overlap", type=int, default=1, help="Number of overlapping frames between clips.")

    args = parser.parse_args()

    main(
        video_dir=args.video_dir,
        metadata_dir=args.metadata_dir,
        question_dir=args.question_dir,
        question_files=args.question_files,
        video_ids=args.video_ids,
        out_dir_root=args.out_dir_root,
        keep_every=args.keep_every,
        trim_video=args.trim_video,
        trim_delta_sec=args.trim_delta_sec,
        clip_ends=args.clip_ends,
        overwrite=args.overwrite,
        dryrun=args.dryrun,
        debug=args.debug,
        num_workers=args.num_workers,
        max_frames_in_clip=args.max_frames_in_clip,
        overlap=args.overlap,
    )
