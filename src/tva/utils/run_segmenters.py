#!/usr/bin/env python3
import os
import sys
import time
import argparse
import subprocess
import glob
import yaml
import pathlib
import shutil
import tempfile
import uuid
from collections import deque

def parse_args():

    """
    # Example usage:
    python3 run_segmenters.py --script ./cache_video_segments.py --question-dir data/questions/yamls --gpus 4,5,6,7 --tmp-root data/videos/subsampled/subspl-by-4/ --dump-dir tmp/tva_vid_segs/subspl-by-4  --jobs-per-gpu 1 --video-dir data/videos/subsampled/subspl-by-4/ 
    """


    p = argparse.ArgumentParser(description="Launch concurrent SAM2 video-seg jobs across available GPU slots.")
    p.add_argument("--script", type=str, required=True,
                   help="Path to your segmentation script (patched with CLI args). Example: segment_videos.py")
    p.add_argument("--video-dir", type=str, required=True,
                   help="Directory containing videos to segment.")
    p.add_argument("--question-dir", type=str, required=True,
                   help="Directory containing *.yaml question files.")
    p.add_argument("--gpus", type=str, default="0,1,2,3",
                   help="Comma-separated GPU ids to use, e.g. '0,1,2,3'.")
    p.add_argument("--jobs-per-gpu", type=int, default=1,
                   help="Number of concurrent jobs allowed on each GPU.")
    p.add_argument("--include", type=str, nargs="+", default=["*.yaml"],
                   help="Glob(s) or explicit filenames to select question files. Example: '*.yaml' or 'file1.yaml file2.yaml'.")
    p.add_argument("--exclude", type=str, default=None,
                   help="Optional glob to exclude files.")
    p.add_argument("--overwrite-existing", action="store_true",
                   help="Forward to underlying script.")
    p.add_argument("--overwrite-cache", action="store_true",
                   help="Forward to underlying script.")
    p.add_argument("--extra-args", type=str, default="",
                   help="Extra CLI args to forward, e.g. '--debug-stride 5'.")
    p.add_argument("--logs-dir", type=str, default="seg_logs",
                   help="Directory to write per-job logs.")
    p.add_argument("--python", type=str, default=sys.executable,
                   help="Python executable to use for subprocesses.")
    p.add_argument("--poll-seconds", type=float, default=2.0,
                   help="Polling interval to check finished jobs.")
    p.add_argument("--tmp-root", type=str, default="/tmp",
                   help="Root directory under which per-job cache dirs are created.")
    p.add_argument("--dump-dir", type=str, default=None, required=True,
                   help="Directory to dump processed video segments. Default is set based on video_dir.")
    p.add_argument("--cleanup-cache", action="store_true",
                   help="Remove per-job cache directories when jobs finish.")
    p.add_argument("--is-trimmed-video", action="store_true",
                   help="Whether the input videos are already trimmed to relevant segments.")
    return p.parse_args()

def make_temp_cache(tmp_root, add_uuid=False):
    """Create a unique per-job cache directory."""
    cache_dir = tmp_root
    if add_uuid:
        cache_dir = os.path.join(tmp_root, f"tva_cache_job_{uuid.uuid4().hex[:8]}")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def safe_rmtree(path):
    try:
        shutil.rmtree(path)
    except Exception as e:
        print(f"[WARN] Failed to remove {path}: {e}")

def main():
    args = parse_args()
    logs_dir = pathlib.Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_ids:
        print("No GPUs specified.")
        sys.exit(1)
    if args.jobs_per_gpu < 1:
        print("--jobs-per-gpu must be >= 1.")
        sys.exit(1)
    gpu_slots = [
        gpu_id
        for gpu_id in gpu_ids
        for _ in range(args.jobs_per_gpu)
    ]
    if not gpu_slots:
        print("No GPU slots available.")
        sys.exit(1)

    include_entries = args.include if isinstance(args.include, list) else [args.include]
    seen_candidates = set()
    all_candidates = []
    for include_entry in include_entries:
        if os.path.isabs(include_entry):
            candidate_path = include_entry
            is_file = os.path.isfile(candidate_path)
            matches = [candidate_path] if is_file else sorted(glob.glob(candidate_path))
        else:
            candidate_path = os.path.join(args.question_dir, include_entry)
            is_file = os.path.isfile(candidate_path)
            matches = [candidate_path] if is_file else sorted(glob.glob(candidate_path))
        for match in matches:
            if match not in seen_candidates:
                seen_candidates.add(match)
                all_candidates.append(match)
    if args.exclude:
        excl = set(glob.glob(os.path.join(args.question_dir, args.exclude)))
        question_files = [q for q in all_candidates if q not in excl]
    else:
        question_files = all_candidates

    qfiles = [os.path.basename(q) for q in question_files]
    
    # ensure that only qfiles with unique prompts are retained
    uniq_qfiles = []
    seen_prompts = set()
    for qfile in qfiles:
        with open(os.path.join(args.question_dir, qfile), "r") as f:
            question_json = yaml.safe_load(f)
        video_id = question_json.get("video_id", "")
        furniture_category = question_json.get("category", "")
        video_name = question_json.get("name", "")
        frame_idxs = question_json.get("frame_idx", [])
        if isinstance(frame_idxs, int):
            frame_idxs = [frame_idxs]
        for frame_idx in frame_idxs:
            prompt = f"{video_id}_{frame_idx}"
            if prompt not in seen_prompts:
                seen_prompts.add(prompt)
                uniq_qfiles.append([qfile, furniture_category, video_name, video_id])

    jobs = deque([(os.path.basename(q[0]), *q[1:]) for q in uniq_qfiles])
    running = {}
    free_gpu_pool = deque(gpu_slots)

    def launch_one(gpu_id: str, qfile_basename: str, furniture_category: str, video_name: str, video_id: str):
        cache_dir = make_temp_cache(
            os.path.join(args.tmp_root, furniture_category, video_name, video_id, "clips")
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id

        log_path = logs_dir / f"{qfile_basename.replace('.yaml','')}.gpu{gpu_id}.log"
        if args.is_trimmed_video:
            log_path = log_path.with_name(log_path.name.replace("gpu", "gpu_trimmed"))
        log_fh = open(log_path, "w", buffering=1)
        if args.is_trimmed_video:
            log_fh.write("is_trimmed=True\n")
            log_fh.flush()

        cmd = [
            args.python,
            args.script,
            "--video-dir", args.video_dir,
            "--filter-add", qfile_basename,
            "--device", "cuda",
            "--cache-dir", cache_dir,
            "--dump-dir", args.dump_dir,
        ]
        if args.overwrite_existing:
            cmd.append("--overwrite-existing")
        if args.overwrite_cache:
            cmd.append("--overwrite-cache")
        if args.is_trimmed_video:
            cmd.append("--is-trimmed-video-dir")
        if args.extra_args.strip():
            cmd.extend(args.extra_args.strip().split())

        print(f"[LAUNCH] GPU {gpu_id} -> {qfile_basename}, cache={cache_dir}")
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
        )
        return {
            "proc": proc,
            "gpu": gpu_id,
            "qfile": qfile_basename,
            "cache_dir": cache_dir,
            "log_fh": log_fh,
            "log_path": str(log_path),
        }

    # Prime initial launches
    while free_gpu_pool and jobs:
        gpu = free_gpu_pool.popleft()
        qfile = jobs.popleft()
        info = launch_one(gpu, *qfile)
        running[info["proc"].pid] = info

    while running or jobs:
        finished_pids = []
        for pid, info in list(running.items()):
            ret = info["proc"].poll()
            if ret is not None:
                finished_pids.append(pid)
                info["log_fh"].close()
                status = "OK" if ret == 0 else f"EXIT={ret}"
                print(f"[DONE] GPU {info['gpu']} <- {info['qfile']} [{status}] log={info['log_path']}")
                if args.cleanup_cache:
                    safe_rmtree(info["cache_dir"])
                else:
                    print(f"[CACHE] Preserved {info['cache_dir']}")
                free_gpu_pool.append(info["gpu"])

        for pid in finished_pids:
            running.pop(pid, None)

        while free_gpu_pool and jobs:
            gpu = free_gpu_pool.popleft()
            qfile = jobs.popleft()
            info = launch_one(gpu, *qfile)
            running[info["proc"].pid] = info

        time.sleep(args.poll_seconds)

    print("All jobs complete.")

if __name__ == "__main__":
    main()
