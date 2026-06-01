import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    pythonpath=True,
    dotenv=True,
)
from typing import Callable, Optional, Tuple, Union, List, Literal
import json
import tempfile
from pathlib import Path
from collections import deque
import subprocess, shlex

import av
import numpy as np
import mediapy as media
from pathlib import Path

from fractions import Fraction
from src.tva.utils.common import _fraction_to_float

EPS = 1e-9

def _merge_and_invert(remove_ranges, total):
    # normalize, clamp, sort, merge
    rs = []
    for a, b in remove_ranges:
        s, e = (float(a), float(b)) if a <= b else (float(b), float(a))
        s = max(0.0, min(s, total))
        e = max(0.0, min(e, total))
        if e - s > EPS:
            rs.append((s, e))
    rs.sort()
    merged = []
    for s, e in rs:
        if not merged or s > merged[-1][1] + EPS:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    # invert -> keep
    keep, t = [], 0.0
    for s, e in merged:
        if s > t: keep.append((t, s))
        t = e
    if t < total: keep.append((t, total))
    return [(s, e) for s, e in keep if e - s > EPS]

def _safe_pick_fps(v_in, nominal_fps=None):
    if nominal_fps is not None:
        return Fraction(nominal_fps).limit_denominator()
    for attr in ("average_rate", "base_rate"):
        r = getattr(v_in, attr, None)
        try:
            if r is not None and float(r) > 0:
                return Fraction(r).limit_denominator()
        except Exception:
            pass
    return Fraction(30, 1)

def cut_out_ranges_pyav_seconds_mp4_safe(
    input_path: str,
    output_path: str,
    remove_ranges: list[tuple[float, float]],
    *,
    # NEW: optional head/tail trims (seconds)
    remove_before: float | None = None,   # removes [0, remove_before)
    remove_after:  float | None = None,   # removes [remove_after, end)
    vcodec: str = "libx264",
    crf: int = 20,
    preset: str = "veryfast",
    pix_fmt: str = "yuv420p",
    nominal_fps: float | None = None,
) -> list[tuple[float, float]]:
    """
    Single-pass PyAV trim/remove that is safe for MP4:
      - decodes once (no seeking), no B-frames, explicit monotone PTS.
      - video-only re-encode.

    remove_before: if set, also remove [0, remove_before)
    remove_after : if set, also remove [remove_after, video_end)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    in_ctn = av.open(str(input_path))
    v_in = next((s for s in in_ctn.streams if s.type == "video"), None)
    if v_in is None:
        in_ctn.close()
        raise RuntimeError("No video stream found.")

    # Duration in seconds
    if in_ctn.duration is not None:
        total_sec = in_ctn.duration / av.time_base  # av.time_base = 1e6
    elif v_in.duration is not None and v_in.time_base is not None:
        total_sec = float(v_in.duration * v_in.time_base)
    else:
        # If unknown, assume a very large end; head/tail trims will still work.
        total_sec = 1e12

    # Build the full removal set, including optional head/tail cuts
    full_remove = list(remove_ranges)
    if remove_before is not None:
        full_remove.append((0.0, float(remove_before)))
    if remove_after is not None:
        full_remove.append((float(remove_after), total_sec))

    keep = _merge_and_invert(full_remove, total_sec)
    if not keep:
        in_ctn.close()
        raise ValueError("After removing the requested ranges, no content remains.")

    # Fast interval membership state
    ki = 0
    kstart, kend = keep[0]
    def in_keep(t):
        nonlocal ki, kstart, kend
        while ki < len(keep) and t > kend + EPS:
            ki += 1
            if ki < len(keep):
                kstart, kend = keep[ki]
        return (ki < len(keep)) and (kstart - EPS <= t < kend - EPS)

    # Output setup
    fps = _safe_pick_fps(v_in, nominal_fps)  # guaranteed Fraction
    out_ctn = av.open(str(output_path), mode="w")
    v_out = out_ctn.add_stream(vcodec, rate=fps)

    if v_out.codec_context.name == "libx264":
        v_out.codec_context.options = {
            **v_out.codec_context.options,
            "preset": preset,
            "crf": str(crf),
        }

    # MP4-safe: disable B-frames so DTS==PTS monotone
    v_out.codec_context.max_b_frames = 0

    v_out.pix_fmt = pix_fmt
    v_out.width  = v_in.codec_context.width
    v_out.height = v_in.codec_context.height

    out_tb = Fraction(1, int(round(float(fps))))   # e.g., 1/30
    v_out.time_base = out_tb
    v_out.codec_context.time_base = out_tb
    v_out.codec_context.framerate = fps

    v_tb_in = v_in.time_base or Fraction(1, 1000)

    def _encode_and_mux(frame=None):
        for pkt in v_out.encode(None if frame is None else frame):
            out_ctn.mux(pkt)

    next_pts = 0  # explicit, strictly increasing PTS in out_tb units

    try:
        # Single-pass decode in order; no seeks
        for frame in in_ctn.decode(video=v_in.index):
            if frame.pts is None:
                continue
            t_sec = float(frame.pts * v_tb_in)
            if not in_keep(t_sec):
                continue

            # Assign monotone PTS; encoder outputs DTS==PTS (no B-frames)
            frame.pts = next_pts
            frame.time_base = out_tb
            next_pts += 1

            _encode_and_mux(frame)

        _encode_and_mux(None)  # final flush
    finally:
        out_ctn.close()
        in_ctn.close()

    return keep

def get_video_resolution_pyav(path):
    """
    Returns resolution without decoding frames.

    Output dict:
      coded_w, coded_h: stored raster
      rotation_deg: 0/90/180/270 if present
      sar: sample aspect ratio as "num:den" or None (None means square pixels)
      display_w, display_h: after applying rotation and SAR
      dar: display aspect ratio (float) if known
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    with av.open(str(p), mode="r") as container:
        # first video stream
        vs = next(s for s in container.streams if s.type == "video")

        # coded size
        w = vs.codec_context.width
        h = vs.codec_context.height

        # rotation (prefer side data if available, else metadata tag)
        rot = 0
        try:
            # Some builds expose a display matrix side data
            for sd in getattr(vs, "side_data", []):
                t = getattr(sd, "type", "") or getattr(sd, "side_data_type", "")
                if str(t).lower().replace("_", " ") in ("display matrix", "displaymatrix"):
                    rot = (int(round(float(getattr(sd, "rotation", 0)))) % 360)
                    break
        except Exception:
            pass
        if rot == 0:
            rtag = (vs.metadata or {}).get("rotate")
            if rtag is not None:
                try:
                    rot = int(round(float(rtag))) % 360
                except Exception:
                    pass

        # pixel aspect (SAR)
        # Prefer stream.sample_aspect_ratio, else codec_context.sample_aspect_ratio
        sar_frac = getattr(vs, "sample_aspect_ratio", None) or getattr(vs.codec_context, "sample_aspect_ratio", None)
        sar_float = _fraction_to_float(sar_frac)
        sar_str = None
        if sar_float is not None and sar_float > 0 and abs(sar_float - 1.0) > 1e-6:
            # Try to keep exact "num:den" if Fraction present
            if isinstance(sar_frac, Fraction):
                sar_str = f"{sar_frac.numerator}:{sar_frac.denominator}"
            else:
                sar_str = f"{sar_float:.6g}:1"  # approximate

        # start from coded size, then apply rotation
        disp_w, disp_h = (h, w) if (rot % 180 == 90) else (w, h)

        # apply SAR to width (common convention: display_w = width * SAR)
        if sar_float and sar_float > 0 and abs(sar_float - 1.0) > 1e-6:
            disp_w = int(round(disp_w * sar_float))

        dar = disp_w / disp_h if disp_h else None

        return {
            "coded_w": int(w),
            "coded_h": int(h),
            "rotation_deg": int(rot),
            "sar": sar_str,                 # None means square pixels (1:1)
            "display_w": int(disp_w),
            "display_h": int(disp_h),
            "dar": float(dar) if dar else None,
        }

def frames_between_vfr(
    video_path: str,
    start_time: float,
    end_time: Optional[float] = None,
    delta: Optional[float] = None,
    return_fps: bool = False,
    only_return_frame_timestamps: bool = False,
) -> Union[
    np.ndarray,
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, float]
]:
    """
    Extract RGB frames between timestamps from a (possibly variable-FPS) MP4,
    returning frames and per-frame timestamps.

    Args:
        video_path (str): Path to the video.
        start_time (float): Start time in seconds (inclusive).
        end_time (float, optional): End time in seconds (exclusive). Mutually exclusive with `delta`.
        delta (float, optional): Duration in seconds from `start_time`. Mutually exclusive with `end_time`.
        return_fps (bool): If True, also return average FPS over [start_time, end_time).
        only_return_frame_timestamps (bool): If True, skip decoding frames and only return timestamps.

    Returns:
        (frames, timestamps) or (frames, timestamps, avg_fps):
            frames:     np.ndarray, shape (T, H, W, 3), dtype=uint8, RGB
            timestamps: np.ndarray, shape (T,), dtype=float64, seconds
            avg_fps:    float, len(frames) / (end_time - start_time)  (only if return_fps=True)

    Notes:
        - Uses true frame timestamps (PTS * time_base), so works correctly on VFR sources.
        - Interval is half-open: [start_time, end_time). Frames with ts == end_time are excluded.
        - If no frames fall inside the interval, returns empty arrays with shapes (0, 0, 0, 3) and (0,).
        - We seek to just before `start_time` for efficiency, then decode forward.
    """
    # Validate mutually exclusive duration arguments
    # import ipdb; ipdb.set_trace()
    if (end_time is not None) and (delta is not None):  # both provided or both missing
        raise ValueError("Provide exactly one of `end_time` or `delta`.")
    if delta is not None:
        end_time = start_time + float(delta)

    # Sanity on ordering
    if end_time is not None and end_time <= start_time:
        empty_frames = np.empty((0, 0, 0, 3), dtype=np.uint8)
        empty_ts = np.empty((0,), dtype=np.float64)
        return (empty_frames, empty_ts, 0.0) if return_fps else (empty_frames, empty_ts)

    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        tb = float(stream.time_base)  # seconds per PTS tick

        # Efficient seek: go to the closest keyframe at/before start_time
        # Convert seconds -> pts units: pts = time / time_base
        seek_pts = int(start_time / tb) if tb > 0 else None
        if seek_pts is not None:
            # Seek backward to ensure we can decode from a keyframe
            container.seek(seek_pts, stream=stream, any_frame=False, backward=True)

        frames_list = []
        ts_list = []

        for frame in container.decode(stream):
            # Robust timestamp extraction
            # Prefer frame.time (seconds) if present; else use PTS * time_base
            if frame.time is not None:
                ts = float(frame.time)
            else:
                if frame.pts is None or tb <= 0:
                    # Skip frames without a valid timestamp
                    continue
                ts = float(frame.pts * stream.time_base)

            # Skip until we reach start of interval
            if ts < start_time:
                continue
            # Stop when we pass the end (half-open interval)
            if end_time is not None and ts >= end_time:
                break

            # Convert to RGB ndarray (H, W, 3), uint8
            rgb = frame.to_ndarray(format="rgb24")
            if not only_return_frame_timestamps:
                frames_list.append(rgb)
            ts_list.append(ts)

        if frames_list:
            frames_arr = np.stack(frames_list, axis=0)
            ts_arr = np.asarray(ts_list, dtype=np.float64)
        elif ts_list:
            ts_arr = np.asarray(ts_list, dtype=np.float64)
        else:
            frames_arr = np.empty((0, 0, 0, 3), dtype=np.uint8)
            ts_arr = np.empty((0,), dtype=np.float64)

    finally:
        container.close()

    if return_fps:
        duration = max(1e-12, end_time - start_time)
        avg_fps = float(len(ts_arr)) / duration
        return frames_arr, ts_arr, avg_fps
    elif only_return_frame_timestamps:
        return ts_arr
    return frames_arr, ts_arr


def find_frame_indices(
    timestamps: np.ndarray,
    query_time: Union[float, np.ndarray, list],
    mode: str = "nearest"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Map query times (in seconds) to frame indices in a variable-FPS video.

    Args:
        timestamps (np.ndarray): Array of shape (T,) with per-frame timestamps (seconds).
        query_time (float | list | np.ndarray): Query time(s) in seconds.
        mode (str): "nearest" or "floor"
            - "nearest": return frame closest to the query time
            - "floor":   return last frame <= query time (like playback)

    Returns:
        (indices, matched_times):
            indices (np.ndarray): Frame indices (shape (N,))
            matched_times (np.ndarray): Actual timestamps of the selected frames (shape (N,))

        If query_time is a float, returns arrays of shape (1,) — you can take [0].
    """
    if len(timestamps) == 0:
        raise ValueError("Empty timestamp array.")

    # Normalize query_time to ndarray
    q = np.atleast_1d(np.array(query_time, dtype=float))

    if mode == "nearest":
        # Compute pairwise distance: shape (Nq, T) → argmin along axis 1
        diff = np.abs(q[:, None] - timestamps[None, :])
        idx = np.argmin(diff, axis=1)

    elif mode == "floor":
        # For each query, find last timestamp <= query
        idx = np.searchsorted(timestamps, q, side="right") - 1
        idx = np.clip(idx, 0, len(timestamps) - 1)

    elif mode == "ceil":
        # For each query, find first timestamp >= query
        idx = np.searchsorted(timestamps, q, side="left")
        idx = np.clip(idx, 0, len(timestamps) - 1)

    else:
        raise ValueError("mode must be 'nearest', 'floor', or 'ceil'.")

    matched_ts = timestamps[idx]

    return idx, matched_ts

def save_rgb_clip_to_temp_mp4(
    frames: np.ndarray,
    fps: float=1,
) -> str:
    """
    Save an RGB video array (T, H, W, 3) to a temporary MP4 and return the path.

    Args:
        frames: np.ndarray with shape (T, H, W, 3), RGB.
                Supports dtype uint8 in [0,255] or float in [0,1].
        fps: frames per second for the output video.
        codec: video codec for mediapy (e.g., "h264", "vp9", "av1").
        quality: encoder quality hint used by mediapy.

    Returns:
        str: Path to the created temporary .mp4 file.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("Expected frames with shape (T, H, W, 3).")

    # Convert to uint8 RGB
    if frames.dtype == np.uint8:
        frames_u8 = frames
    else:
        # Assume floats; handle either [0,1] or [0,255]-ish gracefully
        f = np.asarray(frames, dtype=np.float32)
        if f.max() <= 1.0:
            f = f * 255.0
        frames_u8 = np.clip(np.rint(f), 0, 255).astype(np.uint8)

    # Create a temp file path (not deleted on close so mediapy can write to it)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmp.name
    tmp.close()

    # Write video; mediapy expects RGB order
    media.write_video(tmp_path, frames_u8, fps=fps)

    return tmp_path

def split_video_by_frames(
    input_path: str,
    out_dir: str,
    max_frames: int = 256,
    vcodec: str = "libx264",
    crf: int = 20,
    preset: str = "veryfast",
    nominal_fps: Optional[float] = None,
    overlap: int = 0,
) -> int:
    """
    Split a video into clips of at most `max_frames` frames using streaming decode/encode.
    Outputs clip_0000.mp4, clip_0001.mp4, ... (video-only).

    New:
      - `overlap` (int, default 0): number of frames to overlap between consecutive clips.
        The last `overlap` frames of clip i are duplicated as the first `overlap` frames of clip i+1.
        For safety, the effective overlap is clamped to max(0, min(overlap, max_frames-1)).

    Notes:
      - Output clips are CFR at `nominal_fps` (or input average_rate if available, else 30).
      - Each clip starts PTS at 0.
    """
    if max_frames <= 0:
        raise ValueError("max_frames must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    # Ensure we still make forward progress per clip
    eff_overlap = max(0, min(overlap, max_frames - 1))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_ctn = av.open(input_path)
    v_in = next((s for s in in_ctn.streams if s.type == "video"), None)
    if v_in is None:
        in_ctn.close()
        raise RuntimeError("No video stream found in input.")

    # Choose nominal CFR for output
    if nominal_fps is not None:
        rate = nominal_fps
    elif v_in.average_rate is not None:
        rate = Fraction(v_in.average_rate.numerator, v_in.average_rate.denominator)
    else:
        rate = Fraction(30, 1)

    time_base = Fraction(rate.denominator, rate.numerator)

    def open_new_clip(idx: int):
        """Create a new MP4 container + H.264 stream. Start PTS at 0 for each clip."""
        c = av.open(str(out_dir / f"clip_{idx:04d}.mp4"), mode="w")
        v_out = c.add_stream(vcodec, rate=rate, options={"crf": str(crf), "preset": preset})
        v_out.width = v_in.width
        v_out.height = v_in.height
        v_out.pix_fmt = "yuv420p"
        v_out.time_base = time_base
        return c, v_out, 0  # next_pts

    def encode_and_mux(v_out, enc_frame):
        for pkt in v_out.encode(enc_frame):
            out_ctn.mux(pkt)

    # Rolling buffer of the last `eff_overlap` decoded frames for overlap carryover
    last_frames = deque(maxlen=eff_overlap)

    clip_idx = 0
    out_ctn, v_out, next_pts = open_new_clip(clip_idx)
    frames_in_clip = 0
    total_frames = 0
    num_clips_started = 1 if max_frames > 0 else 0

    try:
        for dec_frame in in_ctn.decode(v_in):
            # If current clip is full, rotate to a new clip and prepend overlap frames
            if frames_in_clip >= max_frames:
                # Flush old encoder & close
                for pkt in v_out.encode():
                    out_ctn.mux(pkt)
                out_ctn.close()

                clip_idx += 1
                out_ctn, v_out, next_pts = open_new_clip(clip_idx)
                frames_in_clip = 0
                num_clips_started += 1

                # Prepend the overlap frames from previous clip (in-order)
                if eff_overlap > 0 and len(last_frames) > 0:
                    # only use as many as we actually had in the previous clip
                    carry = list(last_frames)[-eff_overlap:]
                    for fr in carry:
                        enc_fr = fr.reformat(format="yuv420p", width=v_out.width, height=v_out.height)
                        enc_fr.time_base = time_base
                        enc_fr.pts = next_pts
                        next_pts += 1
                        encode_and_mux(v_out, enc_fr)
                        frames_in_clip += 1
                        # now this clip's "last_frames" should reflect frames we've written
                        last_frames.append(fr)

            # Encode the current (new) frame into the current clip
            enc_frame = dec_frame.reformat(format="yuv420p", width=v_out.width, height=v_out.height)
            enc_frame.time_base = time_base
            enc_frame.pts = next_pts
            next_pts += 1
            encode_and_mux(v_out, enc_frame)

            frames_in_clip += 1
            total_frames += 1

            # Update rolling overlap buffer with the *decoded* frame
            if eff_overlap > 0:
                last_frames.append(dec_frame)

        # Flush last encoder and close
        for pkt in v_out.encode():
            out_ctn.mux(pkt)
        out_ctn.close()
        in_ctn.close()

    except Exception:
        # Best-effort cleanup
        try:
            for pkt in v_out.encode():
                out_ctn.mux(pkt)
        except Exception:
            pass
        try:
            out_ctn.close()
        except Exception:
            pass
        try:
            in_ctn.close()
        except Exception:
            pass
        raise

    # If no frames decoded, remove the empty file (if created)
    if total_frames == 0:
        p = out_dir / f"clip_{clip_idx:04d}.mp4"
        if p.exists() and p.stat().st_size == 0:
            p.unlink()
        return 0

    return num_clips_started

Mode = Literal["metadata", "scan"]

def _frames_via_metadata(input_path: str, nominal_fps: Optional[float]) -> Optional[int]:
    # Try PyAV first
    try:
        with av.open(input_path) as ctn:
            vs = next((s for s in ctn.streams if s.type == "video"), None)
            if vs is None:
                return 0
            if isinstance(vs.frames, int) and vs.frames > 0:
                return int(vs.frames)

            # Estimate via duration × fps
            if nominal_fps is not None:
                fps = float(nominal_fps)
            elif vs.average_rate is not None:
                fps = float(Fraction(vs.average_rate.numerator, vs.average_rate.denominator))
            else:
                fps = 30.0

            seconds = None
            if vs.duration is not None and vs.time_base is not None:
                seconds = float(vs.duration * vs.time_base)
            elif ctn.duration is not None:
                seconds = float(ctn.duration) / 1_000_000.0

            if seconds is not None and seconds > 0:
                return max(0, int(round(seconds * fps)))
    except Exception:
        pass

    # ffprobe fallback
    try:
        cmd = (
            'ffprobe -v error -count_frames -select_streams v:0 '
            '-show_entries stream=nb_read_frames '
            '-of default=nokey=1:noprint_wrappers=1 '
            + shlex.quote(input_path)
        )
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode().strip()
        if out and out.isdigit():
            return int(out)

        cmd = (
            'ffprobe -v error -select_streams v:0 '
            '-show_entries stream=duration,r_frame_rate '
            '-of default=nokey=1:noprint_wrappers=1 '
            + shlex.quote(input_path)
        )
        lines = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode().strip().splitlines()
        if len(lines) >= 2:
            duration = float(lines[0])
            num, den = lines[1].split('/')
            fps = float(num) / float(den)
            return max(0, int(round(duration * fps)))
    except Exception:
        pass

    return None

def _frames_via_streaming_scan(input_path: str) -> int:
    """
    Exact frame count with a streaming decode: we decode frames and drop them immediately.
    Memory is O(1); CPU is proportional to video length. Works for VFR, B-frames, etc.
    """
    total = 0
    with av.open(input_path) as ctn:
        vs = next((s for s in ctn.streams if s.type == "video"), None)
        if vs is None:
            return 0
        for frame in ctn.decode(vs):
            total += 1
    return total

def predict_num_clips_vfr_safe(
    input_path: str,
    max_frames: int = 256,
    overlap: int = 0,
    nominal_fps: Optional[float] = None,
    mode: Mode = "metadata",
) -> int:
    """
    Predict how many clips split_video_by_frames() would produce.
    - mode="metadata": fastest, uses headers (may be approximate for VFR)
    - mode="scan": streaming decode & count (exact for VFR), still low-memory
    """
    if max_frames <= 0:
        raise ValueError("max_frames must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    eff_overlap = max(0, min(overlap, max_frames - 1))
    step = max_frames - eff_overlap  # >= 1

    if mode == "metadata":
        total_frames = _frames_via_metadata(input_path, nominal_fps)
        if total_frames is None:
            # If metadata fails, fall back to a streaming scan to be correct
            total_frames = _frames_via_streaming_scan(input_path)
    else:  # mode == "scan"
        total_frames = _frames_via_streaming_scan(input_path)

    if total_frames <= 0:
        return 0

    return (total_frames + step - 1) // step


def get_subspl_frame_idx_raw_frame_time(
    metadata_fn: str,
    subspl_frame_idx: int = None,
):
    """Get the frame time of the subsampled video frame index
    in the raw video using the metadata file.

    Args:
        metadata_fn (str): Path to the metadata file.
        subspl_frame_idx (int): The subsampled frame index.

    Returns:
        float: The frame time in seconds, or -1 if not found.
    """
    all_frame_times = list()
    with open(metadata_fn, 'r') as f:
        for line_idx, line in enumerate(f):
            
            if subspl_frame_idx is not None and line_idx == subspl_frame_idx:
                meta_json = json.loads(line)
                return meta_json['frame_time']
            elif subspl_frame_idx is None:
                meta_json = json.loads(line)
                all_frame_times.append(meta_json['frame_time'])
    if subspl_frame_idx is None:
        return all_frame_times
    return -1

def convert_frame_idx_subspl_to_raw(
    subspl_frame_idx: Union[int, List[int]],
    vid_fn: str,
    metadata_fn: str,
    vid_is_trimmed: bool = False,
):
    """Convert a frame index in the subsampled video to the corresponding
    frame index in the raw video using the metadata file.

    Args:
        subspl_frame_idx (int, List[int]): Frame index in the subsampled video.
        vid_fn (str): Path to the raw video file.
        metadata_fn (str): Path to the metadata file.
        vid_is_trimmed (bool): Whether the video is trimmed. If True, it means that
            all portions of the video where the sub-sampled frame times in the raw 
            video are > 1 second apart have been removed. Also, the portion of the
            video before the time of the first frame and after the time of the last
            frame have been removed. This requires a special handling when converting
            frame indices.
    Returns:
        int or List[int]: Corresponding frame index in the raw video, or -1 if not found.
    """
    subspl_frame_times = list()
    for f_idx in (subspl_frame_idx if isinstance(subspl_frame_idx, list) else [subspl_frame_idx]):
        subspl_frame_times.append(get_subspl_frame_idx_raw_frame_time(
            metadata_fn, f_idx
        ))
        
    
    if vid_is_trimmed:
        all_subspl_frame_times = get_subspl_frame_idx_raw_frame_time(
            metadata_fn, None
        )
        min_time, max_time = all_subspl_frame_times[0], all_subspl_frame_times[-1]
        gap_start_idxs = [i for i in range(0, len(all_subspl_frame_times)-1)
                          if all_subspl_frame_times[i+1] - all_subspl_frame_times[i] > 1.0]
        # print(gap_start_idxs, len(all_subspl_frame_times))
        cum_gap_time = min_time
        # handle the initial trim at the start
        for j in range(len(all_subspl_frame_times)):
            if all_subspl_frame_times[j] >= cum_gap_time:
                all_subspl_frame_times[j] -= cum_gap_time
        
        gap_time = 0 # reset to handle gaps in between
        for i in range(len(gap_start_idxs)):
            gap_start_idx = gap_start_idxs[i]
            gap_start_time = all_subspl_frame_times[gap_start_idx]
            gap_time = (all_subspl_frame_times[gap_start_idx+1] - all_subspl_frame_times[gap_start_idx])
            for j in range(gap_start_idx+1, len(all_subspl_frame_times)):
                if all_subspl_frame_times[j] > gap_start_time:
                    all_subspl_frame_times[j] -= gap_time

        subspl_frame_times = [all_subspl_frame_times[i] for i in range(len(all_subspl_frame_times)) if i in (subspl_frame_idx if isinstance(subspl_frame_idx, list) else [subspl_frame_idx])]
        # print(subspl_frame_times)
        
    all_raw_frame_times = frames_between_vfr(
        vid_fn,
        start_time=0,
        end_time=None,
        return_fps=False,
        only_return_frame_timestamps=True,
    )
    # print(all_raw_frame_times[:50])
    
    frame_idxs_in_raw_video, _ = find_frame_indices(all_raw_frame_times, subspl_frame_times)
    return frame_idxs_in_raw_video if isinstance(subspl_frame_idx, list) else frame_idxs_in_raw_video[0]


if __name__ == "__main__":
    # Example CLI usage:
    #   python split_by_frames.py /path/to/input.mp4 /path/to/output_dir 256
    import sys
    if len(sys.argv) < 3:
        print("Usage: python split_by_frames.py <input.mp4> <out_dir> [max_frames=256]")
        sys.exit(1)
    inp = sys.argv[1]
    out = sys.argv[2]
    mf = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    n = split_video_by_frames(inp, out, max_frames=mf)
    print(f"Wrote {n} clip(s) to {out}")