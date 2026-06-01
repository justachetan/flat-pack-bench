from typing import List, Literal
import os
import copy
import os.path as osp

import numpy as np
import mediapy
import imageio.v2 as iio
from loguru import logger


def subspl_concat_video(
    video_path: str,
    output_dir: str = None,
    num_subspl_frames: int = 32,
    initial_preserve_indices: List[int] = [],
    sampling_strategy: Literal["uniform"] = "uniform",
    out_fps: int = 1
):
    """Subsampling function for concat videos

    Subsamples the original video frames while preserving the visual
    prompt frames at the beginning. 

    Subsampled videos are stored in the `output_dir` directory 

    Args:
        video_path (str): input video path
        output_dir (str): output directory path where video will be dumped. If None, it is dumped in same directory as input.
        num_subspl_frames (int, optional): Total number of frames in the subsampled videos.
            This includes `num_spl_frames - len(initial_preserve_indices)` frames from the original video. Defaults to 32.
        initial_preserve_indices (List[int], optional): Initial number of frames to preserve. Defaults to [].
        sampling_strategy (Literal["uniform"], optional): Sampling strategy to use. Defaults to "uniform".
    Returns:
        str: Path to the subsampled video file.
    """
    
    frames = mediapy.read_video(video_path)
    T, H, W, C = frames.shape
    num_frames  = T

    output_fn = osp.basename(video_path)
    output_fn_tag, output_fn_ext = output_fn.split(".")
    output_fn = f"{output_fn_tag}_subspl{num_subspl_frames}_numpreserve{'-'.join(map(str, initial_preserve_indices))}_numframes{num_subspl_frames}.{output_fn_ext}"
    
    if output_dir is None:
        output_dir = osp.dirname(video_path)
    output_path = osp.join(output_dir, output_fn)
    os.makedirs(output_dir, exist_ok=True)
    if osp.exists(output_path):
        logger.info(f"Output video already exists at {output_path}, skipping subsampling.")
        return output_path

    final_indices = None
    # Determine the indices of frames to keep
    if num_subspl_frames < num_frames:
        remaining_frames = [i for i in range(num_frames) if i not in initial_preserve_indices]
        
        if sampling_strategy == "uniform":
            num_frames_to_sample = num_subspl_frames - len(initial_preserve_indices)
            if num_frames_to_sample > 0:
                sampled_indices = np.linspace(0, len(remaining_frames) - 1, num_frames_to_sample, dtype=int)
                sampled_indices = [remaining_frames[i] for i in sampled_indices]
            else:
                sampled_indices = []
        else:
            raise ValueError(f"Unsupported sampling strategy: {sampling_strategy}")
        final_indices = initial_preserve_indices + sampled_indices
    else:
        # If the requested number of subsampled frames is greater than or equal to the original number of frames, keep all frames
        # We can avoid unnecessary copying by directly using the original frames
        return video_path
    
    subsampled_frames = frames[final_indices]
    mediapy.write_video(output_path, subsampled_frames, fps=out_fps)
    logger.info(f"Subsampled video saved at {output_path}")
    return output_path


    