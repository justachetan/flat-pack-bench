import math
from bisect import bisect_left
from typing import Iterable, Optional, Sequence, Set, Union, Tuple, List

import av
# from av.time import rescale_q
from fractions import Fraction



from fractions import Fraction
from bisect import bisect_left
from typing import Iterable, Optional, Sequence, Set, List
import av

def _rescale_q(value: int, src_tb: Fraction, dst_tb: Fraction) -> int:
    """Pure-Python equivalent of av_rescale_q (round-to-nearest)."""
    num = value * src_tb.numerator * dst_tb.denominator
    den = src_tb.denominator * dst_tb.numerator
    return int((num + den // 2) // den) if den else 0

def subsample_video_preserve_timestamps(
    in_path: str,
    out_path: str,
    *,
    subsample_every_n: int = 2,
    keep_timestamps_sec: Optional[Sequence[float]] = None,
    keep_indices: Optional[Iterable[int]] = None,
    timestamp_tolerance_sec: float = 0.005,
    video_stream_index: int = 0,
    encoder: str = "libx264",
    crf: int = 18,
    preset: str = "medium",
    pixel_format: str = "yuv420p",
    strict_monotonic_pts: bool = True,
) -> None:
    if subsample_every_n < 1:
        raise ValueError("subsample_every_n must be >= 1")

    keep_idx_set: Set[int] = set(keep_indices or [])
    keep_ts_sorted: List[float] = sorted(keep_timestamps_sec or [])

    def timestamp_is_kept(t: float) -> bool:
        if not keep_ts_sorted:
            return False
        i = bisect_left(keep_ts_sorted, t)
        if i < len(keep_ts_sorted) and abs(keep_ts_sorted[i] - t) <= timestamp_tolerance_sec:
            return True
        if i > 0 and abs(keep_ts_sorted[i - 1] - t) <= timestamp_tolerance_sec:
            return True
        return False

    def should_keep(index: int, t_sec: float) -> bool:
        return (index % subsample_every_n == 0) or (index in keep_idx_set) or timestamp_is_kept(t_sec)

    in_container = av.open(in_path)
    try:
        in_streams = [s for s in in_container.streams if s.type == "video"]
        if not in_streams:
            raise RuntimeError("No video stream found in input.")
        if video_stream_index >= len(in_streams):
            raise IndexError(f"Requested video_stream_index={video_stream_index}, but only {len(in_streams)} video streams present.")
        in_stream = in_streams[video_stream_index]
        in_tb: Fraction = in_stream.time_base

        out_container = av.open(out_path, mode="w")
        try:
            # Create output stream; we’ll set codec context fields before first encode.
            out_stream = out_container.add_stream(encoder, rate=None)
            # Set encoder options (CRF/preset). Some PyAV builds use codec_context.options.
            try:
                out_stream.options = {"crf": str(crf), "preset": preset}
            except Exception:
                # Fallback for older builds: set via codec_context if available.
                pass

            # We’ll fill these lazily when we see the first kept frame:
            cc = out_stream.codec_context
            # Prefer setting codec context attributes for maximum compatibility.
            cc.pix_fmt = pixel_format
            # We want output timestamps to mirror input spacing:
            cc.time_base = in_tb  # If your build rejects this, we’ll still set frame.time_base each frame.

            wrote_params = False  # we’ll finalize width/height at first kept frame
            last_out_pts: Optional[int] = None
            frame_idx = 0

            for frame in in_container.decode(in_stream):
                if frame.pts is None:
                    frame_idx += 1
                    continue
                t_sec = float(frame.pts * in_tb)

                if not should_keep(frame_idx, t_sec):
                    frame_idx += 1
                    continue

                if not wrote_params:
                    # Set dimensions now that we know them
                    cc.width = frame.width
                    cc.height = frame.height
                    wrote_params = True  # header will be written automatically on first mux

                # Rescale PTS into the output stream’s time base (we set cc.time_base=in_tb)
                out_tb: Fraction = cc.time_base if cc.time_base else in_tb
                out_pts = _rescale_q(frame.pts, in_tb, out_tb)

                if strict_monotonic_pts and last_out_pts is not None and out_pts <= last_out_pts:
                    out_pts = last_out_pts + 1

                # Prepare the frame for the encoder
                frame.time_base = out_tb
                frame.pts = out_pts

                # Convert to the requested encoder pixel format if needed
                enc_frame = frame
                if frame.format.name != pixel_format:
                    enc_frame = frame.reformat(format=pixel_format, width=cc.width, height=cc.height)

                # Encode + mux
                for packet in out_stream.encode(enc_frame):
                    out_container.mux(packet)

                last_out_pts = out_pts
                frame_idx += 1

            # Flush encoder
            for packet in out_stream.encode(None):
                out_container.mux(packet)
        finally:
            out_container.close()
    finally:
        in_container.close()


def _rescale_q_exact(value: int, src_tb: Fraction, dst_tb: Fraction, mode: str = "nearest") -> int:
    """
    Pure-Python av_rescale_q with selectable rounding.
    mode in {"nearest","floor","ceil","trunc"}.
    """
    num = value * src_tb.numerator * dst_tb.denominator
    den = src_tb.denominator * dst_tb.numerator
    if den == 0:
        return 0
    if mode == "nearest":
        return int((num + (den // 2)) // den)
    elif mode == "floor":
        return int(num // den)
    elif mode == "ceil":
        return int(-(-num // den))  # ceil for ints
    elif mode == "trunc":
        return int(num // den) if num >= 0 else int(-((-num) // den))
    else:
        raise ValueError("Invalid mode")

def _secs_to_pts(
    seconds: Union[float, str, Fraction],
    tb: Fraction,
    mode: str = "nearest",
) -> int:
    """
    Convert seconds to PTS (integer in units of tb) exactly, using rationals.
    Pass seconds as str or Fraction to avoid float drift:
      - "1.234"  or  Fraction(1234, 1000)
    """
    # seconds_frac is exact if str or Fraction is provided
    if isinstance(seconds, Fraction):
        seconds_frac = seconds
    elif isinstance(seconds, str):
        seconds_frac = Fraction(seconds)  # exact rational from decimal string
    else:
        # fall back: convert float via string to reduce binary error
        seconds_frac = Fraction(str(seconds))
    # pts = round( seconds / tb )
    # value is measured in src_tb=1 (seconds), map to dst_tb=tb
    # i.e., value=seconds_frac (as rational) * (1/tb)
    # We do it by building an equivalent integer rescale:
    # seconds_frac / tb = seconds_frac * tb_den / tb_num
    num = seconds_frac.numerator * tb.denominator
    den = seconds_frac.denominator * tb.numerator
    if den == 0:
        return 0
    if mode == "nearest":
        return int((num + (den // 2)) // den)
    elif mode == "floor":
        return int(num // den)
    elif mode == "ceil":
        return int(-(-num // den))
    elif mode == "trunc":
        return int(num // den) if num >= 0 else int(-((-num) // den))
    else:
        raise ValueError("Invalid mode")

def subsample_video_preserve_timestamps_exact(
    in_path: str,
    out_path: str,
    *,
    subsample_every_n: int = 2,
    # Exact matching knobs (choose one or both):
    keep_pts: Optional[Iterable[int]] = None,                   # exact input PTS integers
    keep_times: Optional[Iterable[Union[float, str, Fraction]]] = None,  # exact seconds -> PTS via Fraction
    time_rounding: str = "nearest",   # rounding used for seconds->PTS mapping
    video_stream_index: int = 0,
    encoder: str = "libx264",
    crf: int = 18,
    preset: str = "medium",
    pixel_format: str = "yuv420p",
    strict_monotonic_pts: bool = True,  # set False if you require *byte-exact* input PTS in output
) -> None:
    """
    Sub-sample VFR video; keep exact frames by PTS and/or exact timestamps (no tolerance).
    Output frames carry the original input PTS (rescaled only if encoder forces a different time_base).
    """
    if subsample_every_n < 1:
        raise ValueError("subsample_every_n must be >= 1")

    keep_pts_set: Set[int] = set(keep_pts or [])

    in_container = av.open(in_path)
    try:
        in_streams = [s for s in in_container.streams if s.type == "video"]
        if not in_streams:
            raise RuntimeError("No video stream found in input.")
        if video_stream_index >= len(in_streams):
            raise IndexError(f"video_stream_index={video_stream_index}, only {len(in_streams)} video streams present.")
        in_stream = in_streams[video_stream_index]
        in_tb: Fraction = in_stream.time_base

        # If keep_times given, precompute their *exact* input-PTS targets:
        keep_pts_from_times: Set[int] = set()
        if keep_times:
            for t in keep_times:
                keep_pts_from_times.add(_secs_to_pts(t, in_tb, time_rounding))

        out_container = av.open(out_path, mode="w")
        try:
            out_stream = out_container.add_stream(encoder, rate=None)
            try:
                out_stream.options = {"crf": str(crf), "preset": preset}
            except Exception:
                pass

            cc = out_stream.codec_context
            cc.pix_fmt = pixel_format
            # Make output time base match input to avoid any rescale drift
            cc.time_base = in_tb

            wrote_params = False
            last_out_pts = None
            frame_idx = 0

            for frame in in_container.decode(in_stream):
                if frame.pts is None:
                    frame_idx += 1
                    continue

                # Decide keep:
                keep_by_stride = (frame_idx % subsample_every_n == 0)
                keep_by_exact_pts = (frame.pts in keep_pts_set) or (frame.pts in keep_pts_from_times)
                if not (keep_by_stride or keep_by_exact_pts):
                    frame_idx += 1
                    continue

                if not wrote_params:
                    cc.width = frame.width
                    cc.height = frame.height
                    wrote_params = True

                # We want to write *exact* input PTS in output timebase.
                out_tb: Fraction = cc.time_base if cc.time_base else in_tb
                out_pts = frame.pts if out_tb == in_tb else _rescale_q_exact(frame.pts, in_tb, out_tb, "nearest")

                # If you need byte-exact PTS, disable this guard; but you may hit muxer errors on bad sources.
                if strict_monotonic_pts and last_out_pts is not None and out_pts <= last_out_pts:
                    # To preserve validity, nudge; comment these two lines if exactness is paramount.
                    out_pts = last_out_pts + 1

                frame.time_base = out_tb
                frame.pts = out_pts

                enc_frame = frame if frame.format.name == pixel_format else frame.reformat(
                    format=pixel_format, width=cc.width, height=cc.height
                )

                for packet in out_stream.encode(enc_frame):
                    out_container.mux(packet)

                last_out_pts = out_pts
                frame_idx += 1

            for packet in out_stream.encode(None):
                out_container.mux(packet)
        finally:
            out_container.close()
    finally:
        in_container.close()


if __name__ == "__main__":
    import os
    import os.path as osp
    import glob
    import yaml
    import json
    import argparse
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="Sub-sample a video while preserving selected frames and timing.")
    parser.add_argument("vid_dir_root", type=str, help="Root directory containing input videos.")
    parser.add_argument("out_dir_root", type=str, help="Root directory to write output videos.")
    parser.add_argument("question_dir_root", type=str,
                        help="Root directory containing the question YAML files" )
    parser.add_argument("metadata_dir_root", type=str,
                        help="Root directory containing the frame metadata JSONL files.")

    parser.add_argument("--subsample-every-n", type=int, default=2,
                        help="Keep every N-th decoded frame (0-based): i % N == 0 is kept. Set to 1 to keep all frames.")
    parser.add_argument("--timestamp-tolerance-sec", type=float, default=0.005,
                        help="Matching tolerance for timestamp preservation.")
    parser.add_argument("--video-stream-index", type=int, default=0,
                        help="Which video stream to read from (0-based index).")
    parser.add_argument("--encoder", type=str, default="libx264",
                        help="FFmpeg encoder name for the output.")
    parser.add_argument("--crf", type=int, default=18,
                        help="Constant Rate Factor for quality (lower is higher quality).")
    parser.add_argument("--preset", type=str, default="medium",
                        help="Encoder speed/efficiency preset.") 
    parser.add_argument("--pixel-format", type=str, default="yuv420p",
                        help="Pixel format required for broad compatibility.")
    parser.add_argument("--strict-monotonic-pts", action="store_true", default=False
                        , help="Enforce strictly increasing output PTS (fixes muxer errors).")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="If set, only print what would be done without writing output.")
    parser.add_argument("--overwrite", action="store_true", default=False,
                        help="If set, overwrite existing output files.")
    args = parser.parse_args()

    os.makedirs(args.out_dir_root, exist_ok=True)

    # first we collect the frame indices in each video that we need to preserve
    video_to_preserve_indices = {}
    question_paths = glob.glob(osp.join(args.question_dir_root, "*.yaml"))
    print(f"Found {len(question_paths)} question files under {args.question_dir_root}")
    for qpath in question_paths:
        with open(qpath, "r") as f:
            qdata = yaml.safe_load(f)
        video_id = qdata["video_id"]
        frame_indices = qdata["frame_idx"]
        if video_id not in video_to_preserve_indices:
            video_to_preserve_indices[video_id] = set()
        if isinstance(frame_indices, int):
            frame_indices = [frame_indices]
        video_to_preserve_indices[video_id].update(frame_indices)



    video_paths = glob.glob(osp.join(args.vid_dir_root, "*", "*", "*", "*.mp4"))
    print(f"Found {len(video_paths)} videos under {args.vid_dir_root}")

    for in_path in tqdm(video_paths, desc="Processing videos", unit="video", total=len(video_paths)):
        rel_path = osp.relpath(in_path, args.vid_dir_root)
        out_path = osp.join(args.out_dir_root, rel_path)
        meta_path = osp.join(args.metadata_dir_root, rel_path[:-4] + "_frames_metadata.jsonl")
        video_id = osp.basename(in_path)[:-4]
        if video_id not in video_to_preserve_indices:
            print(f"[SKIP] No questions for video {video_id}, skipping.")
            continue

        if osp.exists(out_path) and not args.overwrite:
            print(f"[SKIP] Output exists: {out_path}")
            continue

        os.makedirs(osp.dirname(out_path), exist_ok=True)

        keep_timestamps_sec = None
        if osp.exists(meta_path):
            with open(meta_path, "r") as f:
                for line_idx, line in enumerate(f):
                    if line_idx in video_to_preserve_indices[video_id]:
                        if keep_timestamps_sec is None:
                            keep_timestamps_sec = []

                        meta = json.loads(line.strip())
                        keep_timestamps_sec.append(meta["frame_time"])


            if keep_timestamps_sec is not None:
                print(f"  Loaded {len(keep_timestamps_sec)} timestamps to preserve from {meta_path}")
        else:
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        
        print(f"Processing:\n  IN:  {in_path}\n  OUT: {out_path}")
        if not args.dry_run:
            # subsample_video_preserve_timestamps(
            #     in_path,
            #     out_path,
            #     subsample_every_n=args.subsample_every_n,
            #     keep_timestamps_sec=keep_timestamps_sec,
            #     timestamp_tolerance_sec=args.timestamp_tolerance_sec,
            #     video_stream_index=args.video_stream_index,
            #     encoder=args.encoder,
            #     crf=args.crf,
            #     preset=args.preset,
            #     pixel_format=args.pixel_format,
            #     strict_monotonic_pts=args.strict_monotonic_pts,
            # )
            subsample_video_preserve_timestamps_exact(
                in_path,
                out_path,
                subsample_every_n=args.subsample_every_n,
                keep_times=keep_timestamps_sec,
                time_rounding="nearest",
                video_stream_index=args.video_stream_index,
                encoder=args.encoder,
                crf=args.crf,
                preset=args.preset,
                pixel_format=args.pixel_format,
                strict_monotonic_pts=args.strict_monotonic_pts,
            )