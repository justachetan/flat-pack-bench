from __future__ import annotations
import av
from fractions import Fraction
from pathlib import Path

EPS = 1e-9

def _merge_ranges(ranges):
    """Normalize, clamp (later), sort, and merge."""
    ranges.sort()
    merged = []
    for s, e in ranges:
        if not merged or s > merged[-1][1] + EPS:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    return merged

def _normalize_ranges(remove_ranges, total):
    rs = []
    for a, b in remove_ranges:
        s, e = (float(a), float(b)) if a <= b else (float(b), float(a))
        s = max(0.0, min(s, total))
        e = max(0.0, min(e, total))
        if e - s > EPS:
            rs.append((s, e))
    rs.sort()
    return _merge_ranges(rs)

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

def trim_compact_time_pyav(
    input_path: str,
    output_path: str,
    remove_ranges: list[tuple[float, float]],
    *,
    remove_before: float | None = None,   # also remove [0, remove_before)
    remove_after:  float | None = None,   # also remove [remove_after, end)
    vcodec: str = "libx264",
    crf: int = 20,
    preset: str = "veryfast",
    pix_fmt: str = "yuv420p",
    nominal_fps: float | None = None,     # controls timestamp granularity only
):
    """
    Trim while making the output timeline *continuous*:
      t_out = t_in - total_removed_time_before(t_in).
    Video-only; preserves content order and compacts time (no gaps).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    in_ctn = av.open(str(input_path))
    v_in = next((s for s in in_ctn.streams if s.type == "video"), None)
    if v_in is None:
        in_ctn.close()
        raise RuntimeError("No video stream found.")

    # Total duration (seconds)
    if in_ctn.duration is not None:
        total_sec = in_ctn.duration / av.time_base
    elif v_in.duration is not None and v_in.time_base is not None:
        total_sec = float(v_in.duration * v_in.time_base)
    else:
        total_sec = 1e12  # unknown; just use a huge bound

    # Build merged 'removed' list (with optional head/tail)
    removed = list(remove_ranges)
    if remove_before is not None:
        removed.append((0.0, float(remove_before)))
    if remove_after is not None:
        removed.append((float(remove_after), total_sec))
    removed = _normalize_ranges(removed, total_sec)

    # Precompute cumulative removed time at the start of each removed interval
    # and a running sum for quick lookup:
    # For any t, total_removed_before_t is sum of lengths of removed intervals with end <= t,
    # plus any partial overlap of the current removed interval (not needed here since we only
    # evaluate inside kept regions).
    cum = [0.0]
    starts = []
    ends = []
    for s, e in removed:
        starts.append(s); ends.append(e)
        cum.append(cum[-1] + (e - s))
    # Build 'keep' as complement for membership check
    keep = []
    t = 0.0
    for s, e in removed:
        if s > t: keep.append((t, s))
        t = e
    if t < total_sec: keep.append((t, total_sec))
    keep = [(s, e) for s, e in keep if e - s > EPS]
    if not keep:
        in_ctn.close()
        raise ValueError("After removing the requested ranges, no content remains.")

    # For compact mapping we need: for any keep frame time t in [ks,ke),
    # total_removed_before_t = sum lengths of removed intervals with end <= t.
    # Since t is inside a keep interval, that's exactly the cumulative sum at the
    # last removed interval that ends before ks. Precompute offset per keep segment.
    offsets = []
    r_i = 0
    for ks, ke in keep:
        while r_i < len(removed) and ends[r_i] <= ks + EPS:
            r_i += 1
        offset = cum[r_i]  # total removed strictly before this keep segment
        offsets.append((ks, ke, offset))

    # Output setup
    fps = _safe_pick_fps(v_in, nominal_fps)
    out_tb = Fraction(1, int(round(float(fps))))  # e.g., 1/30 (increase by setting nominal_fps=1000)
    out_ctn = av.open(str(output_path), mode="w")
    v_out = out_ctn.add_stream(vcodec, rate=fps)

    if v_out.codec_context.name == "libx264":
        v_out.codec_context.options = {
            **v_out.codec_context.options,
            "preset": preset,
            "crf": str(crf),
        }
    v_out.codec_context.max_b_frames = 0  # no reordering -> DTS==PTS (MP4-safe)

    v_out.pix_fmt = pix_fmt
    v_out.width  = v_in.codec_context.width
    v_out.height = v_in.codec_context.height

    v_out.time_base = out_tb
    v_out.codec_context.time_base = out_tb
    v_out.codec_context.framerate = fps

    v_tb_in = v_in.time_base or Fraction(1, 1000)

    def _encode_and_mux(frame=None):
        for pkt in v_out.encode(None if frame is None else frame):
            out_ctn.mux(pkt)

    last_pts = -1  # ensure strict monotonicity for muxer
    # iterate through keep windows in order while decoding once
    ki = 0
    ks, ke, offset = offsets[0]

    try:
        for frame in in_ctn.decode(video=v_in.index):
            if frame.pts is None:
                continue
            t_in = float(frame.pts * v_tb_in)

            # advance which keep window we’re in
            while ki < len(offsets) and t_in >= offsets[ki][1] - EPS:
                ki += 1
                if ki < len(offsets):
                    ks, ke, offset = offsets[ki]

            # if not in current keep window, skip
            if ki >= len(offsets) or t_in < ks - EPS or t_in >= ke - EPS:
                continue

            # compacted timeline: subtract total removed BEFORE this keep segment
            t_out = t_in - offset
            pts_out = int(round(t_out / float(out_tb)))
            if pts_out <= last_pts:
                pts_out = last_pts + 1

            frame.pts = pts_out
            frame.time_base = out_tb
            _encode_and_mux(frame)
            last_pts = pts_out

        _encode_and_mux(None)
    finally:
        out_ctn.close()
        in_ctn.close()

    return keep

if __name__ == "__main__":
    import os
    import os.path as osp
    import json
    import glob
    import shutil

    older_video_path = osp.join(root, "data", "videos", "subsampled", "subspl-by-4")
    

    older_video_files = glob.glob(older_video_path + f"/*/*/*/*.mp4")
    for fn in older_video_files:
        all_frame_times = []
        metadata_fn = fn.replace("/videos/subsampled/subspl-by-4/", "/frames-metadata/").replace(".mp4", "_frames_metadata.jsonl")
        if not osp.exists(metadata_fn):
            print(f"Skipping {fn} as no metadata file")
            continue
        with open(metadata_fn, 'r') as f:
            for line in f:
                frame_data = json.loads(line.strip())
                all_frame_times.append(frame_data['frame_time'])

        all_frame_times = sorted(all_frame_times)
        remove_ranges = [(i, j) for i, j in zip(all_frame_times[:-1], all_frame_times[1:]) if j - i > 1.0]
        new_video_fn = osp.join(root, "tmp", "trimmed-videos", *fn.split("/")[-4:])
        
        new_video_dir = osp.dirname(new_video_fn)
        os.makedirs(new_video_dir, exist_ok=True)

        

        print(f"Processing {fn} -> {new_video_fn} with {len(remove_ranges)} large gaps")
        # cut_out_ranges_pyav_seconds_mp4_safe(
        #     input_path=fn, output_path=new_video_fn, remove_ranges=remove_ranges, crf=18,
        #     remove_after=all_frame_times[-1],
        #     remove_before=all_frame_times[0],
        # )
        try:
            # NOTE: even if there are no large gaps,
            # we still want to re-encode to remove the initial
            # and final black clips before the first and last
            # keyframes
            trim_compact_time_pyav(
                input_path=fn, output_path=new_video_fn, remove_ranges=remove_ranges, crf=18,
                remove_after=all_frame_times[-1],
                remove_before=all_frame_times[0],
            )
        except Exception as e:
            with open("./failed_trim_videos.txt", 'a+') as f:
                f.write(f"{fn}\n")
