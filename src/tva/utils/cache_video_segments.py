import pyrootutils
root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=".git",
    pythonpath=True,
    dotenv=True,
)
import argparse

from typing import List, Union, Dict
import os
import os.path as osp

import yaml
import json

import numpy as np
import pandas as pd

from pycocotools.mask import decode, encode

from src.tva.media.video_segment import VideoSegment
from src.tva.utils.video import convert_frame_idx_subspl_to_raw

def encode_mask(mask):
    
    new_mask = mask.copy()
    new_mask = new_mask.astype(np.uint8)
    new_mask = np.asfortranarray(new_mask)
    new_mask = encode(new_mask)
    new_mask["counts"] = new_mask["counts"].decode("utf-8")
    return new_mask

def cache_video_segments_without_cuts(
    video_dir: str,
    cache_dir: str,
    dump_dir: str,
    question_fn: str,
    mask_dir: str,
    metadata_dir: str,
    multi_mask: bool = True,
    vos_model: str = "sam2",
    prompt_mode: str = "mask",
    device: str = "cuda",
    overwrite_existing: bool = False,
    is_trimmed_video_dir: bool = False,
    overwrite_cache: bool = False,
    **kwargs
):
    
    with open(question_fn, 'r') as f:
        question_json = yaml.safe_load(f)
    
    question_category = question_json['question_category']
    
    furniture_category = question_json['category']
    furniture_name = question_json['name']
    frame_idxs_subspl = [question_json['frame_idx']] if isinstance(question_json['frame_idx'], int) else question_json['frame_idx']
    video_id = question_json['video_id']
    
    mask_fn = osp.join(mask_dir, furniture_category, furniture_name, video_id, f"{video_id}.json")
    # video_fn = osp.join(video_dir, furniture_category, furniture_name, video_id, f"{video_id}.mp4")
    video_fn = osp.join(video_dir, furniture_category, furniture_name, video_id, "frames")
    # metadata_fn = osp.join(metadata_dir, furniture_category, furniture_name, video_id, f"{video_id}_frames_metadata.jsonl")
    metadata_fn = osp.join(video_dir, furniture_category, furniture_name, video_id, "subspl_frames_metadata.csv")

    # Convert subsampled frame indices to raw frame indices
    # NOTE: since we are now shifting to subsampled frame indices everywhere
    # we do not need to use "original_frame_idx" from metadata
    metadata_df = pd.read_csv(metadata_fn)
    result_df = metadata_df[metadata_df["is_prompt_frame"]][["subsampled_frame_idx", "keyframe_idx"]]
    mapping = result_df.set_index("keyframe_idx")["subsampled_frame_idx"].to_dict()
    frame_idxs = [mapping[item] for item in frame_idxs_subspl]

    video_segmenter = VideoSegment(
        video_fn=video_fn,
        cache_dir=cache_dir,
        multi_mask=multi_mask,
    )

    # NOTE: since we shifted to frame image directories, this step is not needed
    # frame_idxs = convert_frame_idx_subspl_to_raw(
    #     metadata_fn=metadata_fn,
    #     vid_fn=video_fn,
    #     subspl_frame_idx=frame_idxs_subspl,
    #     vid_is_trimmed=is_trimmed_video_dir,
    # )

    
    with open(mask_fn) as f:
        masks = json.load(f)["manual"]
    # import ipdb; ipdb.set_trace()
    prompt_to_track = {
        str(frame_idxs[i]): {
            str(obj_id): decode(masks[str(frame_idxs_subspl[i])][str(obj_id)])
            for obj_id in masks[str(frame_idxs_subspl[i])].keys()
        }
        for i in range(len(frame_idxs))
    }
    
    dump_path = osp.join(dump_dir, furniture_category, furniture_name, video_id)
    os.makedirs(dump_path, exist_ok=True)
    
    
    for i, frame_idx in enumerate(prompt_to_track.keys()):
        
             
        # Set the cached segment filename so that it can be used to dump 
        # the result without creating the big dictionary in memory
        video_segmenter.cached_seg_path = dump_path    
        
        prompt_to_track_for_frame = prompt_to_track[frame_idx]
        complete_segs = video_segmenter.track_object_segments_in_video(
            prompt_to_track=prompt_to_track_for_frame,
            frame_idx=int(frame_idx),
            subspl_frame_idx=int(frame_idxs_subspl[i]),
            vos_model=vos_model,
            prompt_mode=prompt_mode,
            device=device,
            incremental_dump=True,
            overwrite_existing=overwrite_existing,
            overwrite_cache=overwrite_cache,
            non_overlap_masks=True, # this is for SAM2 post-processing
            **kwargs
        )
        
        

def cache_videos_for_questions(
    video_dir: str,
    cache_dir: str,
    dump_dir: str,
    question_dir: str,
    mask_dir: str,
    metadata_dir: str,
    multi_mask: bool = True,
    vos_model: str = "sam2",
    prompt_mode: str = "mask",
    device: str = "cuda",
    process_with_cuts: bool = False,  # Currently not supported
    overwrite_existing: bool = False,
    filter_add: List[str] = None,
    filter_remove: List[str] = None,
    overwrite_cache: bool = False,
    is_trimmed_video_dir: bool = False,
    **kwargs
):
    """Cache video segments for questions.

    Args:
        video_dir (str): Directory containing video files.
        cache_dir (str): Directory to cache processed video segments.
        dump_dir (str): Directory to dump processed video segments.
        question_dir (str): Directory containing question files.
        mask_dir (str): Directory containing mask files.
        metadata_dir (str): Directory containing metadata files.
        multi_mask (bool, optional): Whether to use multiple masks per object. Defaults to True.
        vos_model (str, optional): Video object segmentation model to use. Defaults to "sam2".
        prompt_mode (str, optional): Prompt mode to use. Defaults to "mask".
        device (str, optional): Device to use for processing. Defaults to "cuda".
        process_with_cuts (bool, optional): Whether to process video with cuts. Defaults to False.
        filter_add (List[str], optional): List of files to include. Defaults to None.
        filter_remove (List[str], optional): List of files to not consider. Defaults to None.

    Raises:
        NotImplementedError: _description_
    """
    
    question_fns = [i for i in os.listdir(question_dir) if i.endswith('.yaml')]
    
    if filter_add is not None and filter_remove is not None:
        raise ValueError("Cannot use both filter_add and filter_remove.")
    if filter_add is not None:
        question_fns = [i for i in question_fns if any([f_add == i for f_add in filter_add])]
    elif filter_remove is not None:
        question_fns = [i for i in question_fns if all([f_remove != i for f_remove in filter_remove])]
    
    for question_fn in question_fns:
        question_fn_full = osp.join(question_dir, question_fn)
        
        
        if not process_with_cuts:
            dump_fn = cache_video_segments_without_cuts(
                video_dir=video_dir,
                cache_dir=cache_dir,
                dump_dir=dump_dir,
                question_fn=question_fn_full,
                mask_dir=mask_dir,
                multi_mask=multi_mask,
                metadata_dir=metadata_dir,
                vos_model=vos_model,
                prompt_mode=prompt_mode,
                device=device,
                overwrite_existing=overwrite_existing,
                overwrite_cache=overwrite_cache,
                is_trimmed_video_dir=is_trimmed_video_dir,
                **kwargs
            )
        else:
            raise NotImplementedError(
                "Processing with cuts is not supported yet.")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter-add", nargs="*", default=None,
                        help="One or more question YAML filenames (e.g., 032.yaml). Exact filename match.")
    parser.add_argument("--device", default="cuda",
                        help='Device string for the VOS model. Use plain "cuda" when combined with CUDA_VISIBLE_DEVICES.')
    parser.add_argument("--overwrite-existing", action="store_true",
                        help="Overwrite existing dumps.")
    parser.add_argument("--overwrite-cache", action="store_true",
                        help="Overwrite existing cache.")
    parser.add_argument("--process-with-cuts", action="store_true",
                        help="(Currently not supported) Kept for future.")
    parser.add_argument("--debug-stride", type=int, default=None,
                        help="Forwarded to kwargs if your code uses it.")
    parser.add_argument("--video-dir", type=str, default=None, help="Directory containing video files.")
    parser.add_argument("--cache-dir", type=str, default=osp.join(root, "tmp", "tva_cache"),
                        help="Directory to use for caching video segments.")
    parser.add_argument("--is-trimmed-video-dir", action="store_true",
                        help="Whether the video_dir contains trimmed videos.")
    parser.add_argument("--dump-dir", type=str, default=None,
                        help="Directory to dump processed video segments. Default is set based on video_dir.")
    args = parser.parse_args()
    
    data_dir = osp.join(root, "data")
    manual_data_dir = osp.join(root, "data")

    video_dir = osp.join(data_dir, "videos")
    video_dir = args.video_dir if args.video_dir is not None else video_dir
    dump_dir = args.dump_dir if args.dump_dir is not None else osp.join(root, "tmp", "tva_video_seg_cache")
    question_dir = osp.join(manual_data_dir, "questions", "yamls")
    mask_dir = osp.join(manual_data_dir, "segmentation-masks")
    metadata_dir = osp.join(data_dir, "frames-metadata")

    multi_mask = False
    vos_model = "sam2"
    prompt_mode = "mask"

    cache_videos_for_questions(
        video_dir=video_dir,
        cache_dir=args.cache_dir,
        dump_dir=dump_dir,
        question_dir=question_dir,
        mask_dir=mask_dir,
        metadata_dir=metadata_dir,
        multi_mask=multi_mask,
        vos_model=vos_model,
        prompt_mode=prompt_mode,
        device=args.device,
        process_with_cuts=args.process_with_cuts,
        overwrite_existing=args.overwrite_existing,
        overwrite_cache=args.overwrite_cache,
        filter_add=args.filter_add,     # <-- key
        filter_remove=None,
        debug_stride=args.debug_stride,
        is_trimmed_video_dir=args.is_trimmed_video_dir,
    )
        
        
    
    
