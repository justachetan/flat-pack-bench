import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from typing import Callable, Literal, Optional, Tuple, Union, Dict, List
import os
import os.path as osp
import glob
import json
import copy
import shutil
import subprocess
import inspect

import torch
import numpy as np
import pandas as pd
import mediapy as media
from pycocotools.mask import encode, decode

from loguru import logger

# from src.tva.models.sam2 import SAM2Wrapper
from src.tva.models import forward
from src.tva.utils.video import (
    get_video_resolution_pyav,
    frames_between_vfr,
    split_video_by_frames,
    predict_num_clips_vfr_safe,
)
from src.tva.utils.jsonl_directory_ra import (
    JsonlDirRA,
    decode_jsonl_line,
)

from src.tva.media.prompts.generate_api_prompt import api_desc
from src.tva.utils.parser import override_from_method_params
from src.tva.utils.common import log_event

def encode_mask(mask):
    new_mask = mask.copy()
    new_mask = new_mask.astype(np.uint8)
    new_mask = np.asfortranarray(new_mask)
    new_mask = encode(new_mask)
    new_mask["counts"] = new_mask["counts"].decode("utf-8")
    return new_mask

def current_func_name():
    import inspect
    return inspect.currentframe().f_back.f_code.co_name

@api_desc(
    description="""
        A container object for a video segment that provides operations such as video object
        segmentation, reversing video clips, etc.
        
        Attributes
        ----------
            video_fn: str
                Path to the video file
            h: int
                Height of the video frames
            w: int
                Width of the video frames
            multi_mask: bool
                Whether to use multi-mask prediction in the VOS model
        
        Methods
        -------
            get_video_clip_between(start_time: float, end_time: float) -> str
                Clips the video from start_time to end_time and returns the path to the clipped video file.
            track_object_segments_in_video(prompt_to_track: Dict[str, np.ndarray], frame_idx: int) -> Dict[str, np.ndarray]
                Tracks the object segments in the entire video given the prompts in a specific frame.
    """,
    export=True,
)
class VideoSegment(object):
    
    MAX_NUM_FRAMES_IN_CLIP = 256
    NUM_FRAME_OVERLAP_BETWEEN_CLIPS = 1
    
    @api_desc(
        description="""
        Initializes the VideoSegment object.
        
        Parameters
        ----------
            video_fn (str): Path to the video file.
        """,
        export=True,
        display_signature="(self, video_fn: str)",
        include_code=False,
    )
    def __init__(
        self,
        video_fn: str, 
        cache_dir: str,
        multi_mask: bool = False,
        cached_seg_path: Optional[str] = None,
        jumbled_frame_idxs: Optional[List[int]] = None,
        jumble_map: Optional[dict] = None,
        subspl_factor: Optional[int] = 1,
        method_params: Optional[dict] = None,
    ):
        """VideoSegment Object that provides operations such as video object
        segmentation, reversing video clips, etc.

        Args:
            video_fn (str): Path to the video file or path to a directory containing video frames.
            cache_dir (str): Directory to store intermediate results.
            multi_mask (bool, optional): Whether to use multi-mask prediction in the VOS model. Defaults to False.
            cached_seg_path (Optional[str], optional): path to cached segmentation results. it will be a directory
                structure as follows:
                <cached_seg_path>/<frame_idx>/clip_<clip_index>.jsonl
                or <cached_seg_path>/raw_<frame_idx>_subspl_<subspl_frame_idx>/clip_<clip_index>.jsonl

                where <frame_idx> is the index of the prompt frame in the video where the prompts are provided,
                <subspl_frame_idx> is the keyframe index of the prompt frame in the keyframe videos,
                and <clip_index> is the index of the clip (0-indexed) in the video. 
                Each <frame_idx> directory will contain segmentation results for that prompt frame index, with each clip containing jsonl 
                files for each clip ("clip_<clip_index>.jsonl"). Defaults to None.
            jumbled_frame_idxs (Optional[List[int]], optional): Used only for retrieving cached video object segmentations by 
                the agent. If the part IDs for specific prompt frames need to be jumbled to match the input question, provide 
                the list of jumbled frame indices here. Defaults to None.
            jumble_map (Optional[dict], optional): A mapping from original part IDs to jumbled part IDs. Defaults to None.
            subspl_factor (Optional[int], optional): The subsampling factor used when generating cached segmentations. Defaults to 1.
                Mostly used for sub-sampling with agent.
            method_params (Optional[dict], optional): Parameter values for methods that override inputs. Defaults to None.
        """
        self.video_fn = video_fn
        self.is_video_dir = osp.isdir(video_fn)
        if not self.is_video_dir:
            self.num_frames = media._get_video_metadata(video_fn).num_frames # NOTE: this can be slow for long videos
            self._video_metadata = get_video_resolution_pyav(video_fn)
            self.h, self.w = self._video_metadata['display_h'], self._video_metadata['display_w']
        else:
            self.num_frames = len(os.listdir(video_fn))
            self.h, self.w = media.read_image(osp.join(self.video_fn, os.listdir(video_fn)[0])).shape[:2]
            video_metadata_fn = osp.join(osp.dirname(video_fn), "subspl_frames_metadata.csv")
            metadata_df = pd.read_csv(video_metadata_fn)
            result_df = metadata_df[metadata_df["is_prompt_frame"]][["subsampled_frame_idx", "keyframe_idx"]]
            self.prompt_frame_idxs_in_frame_dir = result_df["subsampled_frame_idx"].tolist()
            # import ipdb; ipdb.set_trace()
            # mapping_keyframe_idx_to_subspl_idx = result_df.set_index("keyframe_idx")["subsampled_frame_idx"].to_dict()

        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.multi_mask = multi_mask
        self.cached_seg_path = cached_seg_path
        self.method_params = method_params or dict()
        self.subspl_factor = subspl_factor

        self.jumbled_frame_idxs = jumbled_frame_idxs
        self.jumble_map = jumble_map

    @override_from_method_params
    @api_desc(
        description="""
        Returns all frames in the video as a numpy array.
        
        Returns
        -------
            np.ndarray: Array of shape (T, H, W, 3) containing all video frames.
        """,
        export=True,
        display_signature="(self)",
        parent_blurb="""
            Returns all frames in the video as a numpy array.
        """,
        body="""return read_frames(self.video_fn)""",
        include_code=False,
    )
    def return_frames(self) -> np.ndarray:
        """Returns all frames in the video as a numpy array.

        Args:
            frame_idxs_to_retain (Optional[List[int]], optional): List of frame indices to retain,
                even if subsampling is used. Defaults to None.
        Returns:
            np.ndarray: Array of shape (T, H, W, 3) containing all video frames.
        """
        if self.is_video_dir:
            # import ipdb; ipdb.set_trace()
            img_fns = sorted(os.listdir(self.video_fn), key=lambda x: int(osp.basename(x).split(".")[0]))
            frame_idxs_to_retain = getattr(self, "prompt_frame_idxs_in_frame_dir") if hasattr(self, "prompt_frame_idxs_in_frame_dir") else []
            frames = [media.read_image(osp.join(self.video_fn, img_fns[fidx])) \
                      for fidx in range(len(img_fns)) \
                        if fidx % self.subspl_factor == 0 or fidx in frame_idxs_to_retain]
            frames = np.stack(frames, axis=0)
        else:
            # TODO: this branch will not align with cached segments if subsampling was used
            frames = media.read_video(self.video_fn)[::self.subspl_factor]
        return frames
    
    @api_desc(
        description="""
        Clips the video from start_time to end_time and saves it to output_fn.
        
        Parameters
        ----------
            start_time (float): Start time in seconds.
            end_time (float): End time in seconds.
        
        Returns
        -------
            str: Path to the clipped video file.
            
        Examples
        --------
        >>> # Clip the video from 10s to 20s
        >>> def execute_code(video_fn):
        >>>     video_segmenter = VideoSegment(video_fn=video_fn)
        >>>     clip_fn = video_segmenter.get_video_clip_between(start_time=10, end_time=20)
        >>>     return clip_fn
        """,
        export=False,
        display_signature="(self, start_time: float=None, end_time: float=None) -> str",
        body="""return get_video_clip_between(self.video_fn, start_time, end_time)""",
        include_code=False,
    )
    def get_video_clip_between(
        self,
        start_time: float=None,
        end_time: float=None,
        output_fn: Optional[str] = None,
    ) -> str:
        """
        Clips the video from start_time to end_time and saves it to output_fn.

        Args:
            start_time (float): Start time in seconds.
            end_time (float): End time in seconds.
            output_fn (Optional[str], optional): Path to save the clipped video. 
                If None, saves to cache_dir with a default name. Defaults to None.
        """
        if output_fn is None:
            if start_time is None or end_time is None:
                raise ValueError("start_time and end_time must be provided either as arguments or in method_kwargs.")
            output_fn = osp.join(self.cache_dir, f"clip_{start_time}_{end_time}.mp4")

        if start_time is None or end_time is None:
            raise ValueError("start_time and end_time must be provided.")

        os.makedirs(osp.dirname(output_fn), exist_ok=True)
        clip_duration = end_time - start_time
        if clip_duration <= 0:
            raise ValueError("end_time must be greater than start_time.")

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-ss",
            f"{start_time:.6f}",
            "-i",
            self.video_fn,
            "-t",
            f"{clip_duration:.6f}",
            "-c",
            "copy",
            output_fn,
        ]
        result = subprocess.run(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(f"ffmpeg failed to clip video: {stderr}")

        return output_fn

    def _track_video_object_segments_in_clip(
        self,
        video: Union[str, List[str]],
        prompt_to_track: Dict[str, np.ndarray],
        frame_idx: int,
        vos_model: Literal["sam2"] = "sam2",
        prompt_mode: Literal["mask", "point"] = "mask",
        non_overlap_masks: bool = True,
        device: str = "cuda",
    ):
        """
        Video Object Segmentation (VOS) in video clip.

        Parameters:
            video (Union[str, List[str]]):
                Path to the video file or a list of paths to video frames.
            prompt_to_track (Dict[str, Dict[str, np.ndarray]]): 
                A dictionary mapping with structure:
                    |-> object_id (str): prompt (np.ndarray)
                where object_id is an integer value as a string
                and prompt is either a binary mask or a set of points.
                Each prompt can either be a binary mask or a set of points, depending on the `prompt_mode`.
            frame_idx (int): 
                The index of the frame in the video clip where the prompts are provided.
                NOTE (for ac): the frame_idx is the index in the (raw) clip, not in the original video.
            vos_model (Literal[&quot;sam2&quot;], optional): name of the video-object segmentation model to use. Defaults to "sam2".
            prompt_mode (Literal[&quot;mask&quot;, &quot;point&quot;], optional): type of the prompt to track, can be either `points' or `masks'. Defaults to "mask".
            device (str, optional): device to use for inference. Defaults to "cuda".
        Returns:
            Dict[str, List[np.ndarray]]:
                A dictionary mapping object IDs to their corresponding segmentation masks across the video clip.
                Each segmentation mask is a binary numpy array of shape (T, H, W), where H and W are the height and width of the video frames, and T is the number of frames in the video clip.
        """



        assert vos_model in ["sam2"], f"vos_model {vos_model} not supported"
        
        object_ids = list(prompt_to_track.keys())
        # object_ids = sum([list(prompt_to_track[f_idx].keys()) for f_idx in frame_idx], [])
        # object_ids = sorted(list(set(object_ids)), key=lambda x: int(x))
        
        masks = np.array([prompt_to_track[part_id] for part_id in object_ids])
        masks = torch.from_numpy(masks.copy()).float()
        # import ipdb; ipdb.set_trace()
        # NOTE: frame_idxs and part_ids should have the corr. values repeated in the same order as the masks
        frame_idxs = [frame_idx for _ in object_ids]
        outputs = self.forward(
            vos_model, 
            init_args={
                "multi_mask": self.multi_mask,
                "non_overlap_masks": non_overlap_masks
            },
            video_dir=video,
            masks=masks,
            frame_idxs=frame_idxs,
            obj_ids=object_ids,
            mode="masks" if prompt_mode=="mask" else "points",
            offload_video_to_cpu=True,
            track_points_by_first_appearance=False,
        )

        segmentations = [
            {
                obj_id: outputs["prediction"][f_idx][obj_id] for obj_id in outputs["prediction"][f_idx]
            }
            for f_idx in outputs["prediction"]
        ]

        # segmentations = dict()
        # for object_id in object_ids:
        #     if object_id not in segmentations:
        #         segmentations[object_id] = list()
            
        #     segmentations[object_id] = np.stack([outputs["prediction"][f_idx][object_id] for f_idx in outputs["prediction"]])

        # assert len(segmentations) == len(prompt_to_track), \
        #     f"Expected {len(prompt_to_track)} segments, but got {len(segmentations)}"
        # assert all(
        #     segmentations[object_id].shape[1:] == (self.h, self.w)
        #     for object_id in segmentations
        # ), "Segmentation masks have incorrect shape"
        return segmentations
    
    # NOTE: for the API, we assume that the prompt is a binary mask only for now to make it easier for the agent
    @override_from_method_params
    @api_desc(
        description="""
        Tracks the object segments in the entire video given the prompts in a specific frame.
        
        Parameters
        ----------
            prompt_to_track (Dict[str, np.ndarray]):
                A dictionary mapping object IDs (integer values as strings) to their corresponding prompts.
                Each prompt is a binary mask of shape (H, W), and the object IDs are strings representing integers.
            frame_idx (int):
                The index of the frame in the video where the prompts are provided.
        
        Returns
        -------
            List[Dict[str, np.ndarray]]:
                A list of dictionaries mapping object IDs to their corresponding segmentation masks across the entire video.
                Each segmentation mask is a binary numpy array of shape (H, W), where H and W are the height and width of the video frames.
                The list is ordered by frame index, and its length is equal to the number of frames in the video.
        
        Examples
        --------
        >>> # Are Parts 0 and 1 connected in frame 10 of the video?
        >>> def execute_code(video_fn, frame_idx, masks):
        >>>    # Here the input frame_idx is the index of the frame at which the masks are provided
        >>>    video_segmenter = VideoSegment(video_fn=video_fn)
        >>>    complete_sgmts = video_segmenter.track_object_segments_in_video(
        >>>        prompt_to_track=masks,
        >>>        frame_idx=frame_idx,
        >>>    )
        >>>    frames = video_segmenter.return_frames()
        >>>    frame = frames[10]
        >>>    segments_for_frame_10 = complete_sgmts[10]
        >>>    image_patch = ImagePatch(
        >>>        input_image=frame,
        >>>        mask=segments_for_frame_10,
        >>>    )
        >>>    return image_patch.check_part_connectivity(part_id1="0", part_id2="1")
        """,
        export=True,
        parent_blurb="""
            Tracks the object segments in the entire video given the prompts in a specific frame.
        """,
        display_signature="(self, prompt_to_track: Dict[str, np.ndarray], frame_idx: int) -> List[Dict[str, np.ndarray]]",
        body="""return track_object_segments_in_video(self.video_fn, prompt_to_track, frame_idx)""",
    )
    def track_object_segments_in_video(
        self,
        prompt_to_track: Dict[str, np.ndarray],
        frame_idx: int,
        subspl_frame_idx: Optional[int] = None,
        non_overlap_masks: bool = True,
        vos_model: Literal["sam2"] = "sam2",
        prompt_mode: Literal["mask", "point"] = "mask",
        device: str = "cuda",
        debug_stride: Optional[int] = None,
        incremental_dump: bool = False,
        overwrite_existing: bool = False,
        overwrite_cache: bool = False,
        is_frame_idx_subspled: bool = True,
    ):
        """
        Video Object Segmentation (VOS) in the entire video.

        Parameters:
            prompt_to_track (Dict[str, np.ndarray]): 
                A dictionary mapping object IDs (integer values as strings) to their corresponding prompts.
                Each prompt can either be a binary mask or a set of points, depending on the `prompt_mode`.
            frame_idx (int): 
                The index of the frame in the video where the prompts are provided.
                NOTE (for ac): the frame_idx is the index in the raw video, not in the keyframe video in 
                the segmentation annotations. 
                NOTE (for ac): if the input video is a subsampled frame directory, this frame index is the
                index in the subsampled frame directory *ignoring* overlap frames between clips. Overlaps
                are accounted for internally in this function.
            subspl_frame_idx (Optional[int], optional): The index of the frame in the subsampled video
                where the prompts are provided. This is only used for dumping the segmentation results
                to a file that can be matched with the question input. Defaults to None.
            non_overlap_masks (bool, optional): Whether to ensure that the predicted masks do not overlap. Defaults to True.
            vos_model (Literal["sam2", "xmem"], optional): name of the video-object 
                segmentation model to use. Defaults to "sam2".
            prompt_mode (Literal["mask", "point"], optional): type of the prompt to track, 
                can be either `points' or `masks'. Defaults to "mask".
            device (str, optional): device to use for inference. Defaults to "cuda".
            debug_stride (Optional[int], optional): If provided, only process clips within this stride
                from the clip containing the prompt frame. This is useful for debugging to limit memory usage.
                Defaults to None.
            incremental_dump (bool, optional): If True, dumps the segmentation results to the cached_seg_fn
                file incrementally after processing each clip. This is useful for long videos to avoid losing progress.
                Defaults to False.
            overwrite_existing (bool, optional): If True, overwrites the existing cached segment files if it exists.
                Defaults to False.
            overwrite_cache (bool, optional): If True, overwrites the existing cache directory (containing the split video clips) 
                if it exists. Defaults to False.
            is_frame_idx_subspled (bool, optional): If True, indicates that the provided frame_idx is in the subsampled video.
                AFTER applying self.subspl_factor to the video frame dir. USED WITH AGENT ONLY. Defaults to True.

        Returns:
            List[Dict[str, np.ndarray]]:
                A list of dictionaries mapping object IDs to their corresponding segmentation masks across the entire video.
                Each segmentation mask is a binary numpy array of shape (H, W), where H and W are the height and width of the video frames.
                The list is ordered by frame index, and its length is equal to the number of frames in the video.
        """
        if not isinstance(prompt_to_track[list(prompt_to_track.keys())[0]], np.ndarray):
            prompt_to_track = {
                part_id: decode(prompt_to_track[part_id]) for part_id in prompt_to_track
            }

        # import ipdb; ipdb.set_trace()
        if is_frame_idx_subspled:
            # NOTE: written for, and tested with, agent only for now
            # with no subsampling, or in the case of agent, this is
            # with respect to the video frames directory

            num_frames_in_frame_dir = self.num_frames
            if self.subspl_factor is not None and self.subspl_factor > 1:
                # we align frame_idx (which is actually frame_idx in the subsampled 
                # video frames dir) to the frame_idx obtained when the frame directory
                # is further subsampled by self.subspl_factor
                # Basically this is an upsampling step because the seg cache files
                # are generated without subsampling by self.subspl_factor
                # import ipdb; ipdb.set_trace()
                all_frame_idxs = list(range(num_frames_in_frame_dir))
                subspl_frame_idxs = [idx for idx in all_frame_idxs if (idx % self.subspl_factor == 0) or (idx in self.prompt_frame_idxs_in_frame_dir)]
                frame_idx = subspl_frame_idxs[frame_idx]

        if self.cached_seg_path is not None:
            
            cached_seg_path_for_frame = osp.join(
                self.cached_seg_path,
                str(frame_idx)
            )
            if subspl_frame_idx is not None:
                cached_seg_path_for_frame = osp.join(
                    self.cached_seg_path,
                    f"raw_{frame_idx}_subspl_{subspl_frame_idx}"
                )
                if not osp.exists(cached_seg_path_for_frame):
                    # logger.debug(f"Exact cached segmentation path {osp.basename(cached_seg_path_for_frame)} not found, trying fuzzy search...")
                    # logger.debug(f"Trying fuzzy search with provided subspl_frame_idx (keyframe idx): {subspl_frame_idx}")
                    log_event(stage="video_segment", event="track_object_segments_in_video",
                              msg=f"Exact cached segmentation path {osp.basename(cached_seg_path_for_frame)} not found, trying fuzzy search with provided subspl_frame_idx (keyframe idx): {subspl_frame_idx}")
                    file_search = glob.glob(osp.join(self.cached_seg_path, f"raw_*_subspl_{subspl_frame_idx}"))
                    if len(file_search) == 0:
                        # logger.debug(f"No cached segmentation files found with subspl_frame_idx {subspl_frame_idx}. Proceeding without cache.")
                        log_event(stage="video_segment", event="track_object_segments_in_video",
                                  msg=f"No cached segmentation files found with subspl_frame_idx {subspl_frame_idx}. Proceeding without cache.")
                    elif len(file_search) > 1:
                        # logger.debug(f"Multiple cached segmentation files found with subspl_frame_idx {subspl_frame_idx}: {file_search}")
                        log_event(stage="video_segment", event="track_object_segments_in_video",
                                  msg=f"Multiple cached segmentation files found with subspl_frame_idx {subspl_frame_idx}: {file_search}")
                        raise ValueError(f"Multiple cached segmentation files found with subspl_frame_idx {subspl_frame_idx}: {file_search}")
                    else:
                        cached_seg_path_for_frame = file_search[0]
                        # logger.debug(f"Found cached segmentation path: {cached_seg_path_for_frame}")
                        log_event(stage="video_segment", event="track_object_segments_in_video",
                                  msg=f"Found cached segmentation path: {cached_seg_path_for_frame}")
            elif not osp.exists(cached_seg_path_for_frame):
                # logger.debug(f"Exact cached segmentation path {osp.basename(cached_seg_path_for_frame)} not found, trying fuzzy search...")
                log_event(stage="video_segment", event="track_object_segments_in_video",
                          msg=f"Exact cached segmentation path {osp.basename(cached_seg_path_for_frame)} not found, trying fuzzy search...")
                file_search = glob.glob(osp.join(self.cached_seg_path, f"raw_{frame_idx}_subspl_*"))
                if len(file_search) == 0:
                    # logger.debug(f"No cached segmentation files found for frame_idx {frame_idx}. Proceeding without cache.")
                    log_event(stage="video_segment", event="track_object_segments_in_video",
                              msg=f"No cached segmentation files found for frame_idx {frame_idx}. Proceeding without cache.")
                elif len(file_search) > 1:
                    # logger.debug(f"Multiple cached segmentation files found for frame_idx {frame_idx}: {file_search}")
                    log_event(stage="video_segment", event="track_object_segments_in_video",
                              msg=f"Multiple cached segmentation files found for frame_idx {frame_idx}: {file_search}")
                    raise ValueError(f"Multiple cached segmentation files found for frame_idx {frame_idx}: {file_search}")
                else:
                    cached_seg_path_for_frame = file_search[0]
                    # logger.debug(f"Found cached segmentation path: {cached_seg_path_for_frame}")  
                    log_event(stage="video_segment", event="track_object_segments_in_video",
                              msg=f"Found cached segmentation path: {cached_seg_path_for_frame}")


            if osp.exists(cached_seg_path_for_frame) and not overwrite_existing:
                
                load_subspl_factor_for_cached_segs = self.subspl_factor

                if load_subspl_factor_for_cached_segs is not None:
                    # logger.info(f"Loading cached segmentation from {cached_seg_path_for_frame} with subsampling factor {load_subspl_factor_for_cached_segs}")
                    log_event(stage="video_segment", event="track_object_segments_in_video",
                                msg=f"Loading cached segmentation from {cached_seg_path_for_frame} with subsampling factor {load_subspl_factor_for_cached_segs}")
                else:
                    load_subspl_factor_for_cached_segs = 1
                    # logger.info(f"Loading cached segmentation from {cached_seg_path_for_frame} ")
                    log_event(stage="video_segment", event="track_object_segments_in_video",
                                msg=f"Loading cached segmentation from {cached_seg_path_for_frame} ")
                
                process_fn = lambda rec: decode_jsonl_line(rec)["masks"]
                if frame_idx in (self.jumbled_frame_idxs or []):
                    process_fn = lambda rec: decode_jsonl_line(rec, jumble_map=self.jumble_map, decode_rle=True)["masks"]

                cached_video_segments_index = JsonlDirRA(
                    dir_path=cached_seg_path_for_frame,
                    skip=load_subspl_factor_for_cached_segs,
                    process=process_fn,
                    retain_indices=self.prompt_frame_idxs_in_frame_dir if hasattr(self, "prompt_frame_idxs_in_frame_dir") else None,
                )

                is_complete_dump = True
                # import ipdb; ipdb.set_trace()
                num_masks = sum(cached_video_segments_index.file_line_counts)
                if num_masks != self.num_frames:
                    # logger.warning(f"Cached segmentation file {cached_seg_path_for_frame} has only {num_masks} frames. The video has {self.num_frames} frames. This indicates that the segmentation was not fully dumped. Proceeding without cache.")
                    log_event(stage="video_segment", event="track_object_segments_in_video",
                                msg=f"cached segmentation file {cached_seg_path_for_frame} has only {num_masks} frames, but the video has {self.num_frames} frames. this indicates that the segmentation was not fully dumped. proceeding without cache.")
                    is_complete_dump = False
                # for idx, item in enumerate(cached_video_segments_index.file_line_counts):
                #     clip_vid = media.read_video(all_clip_fns[idx])
                #     if item < clip_vid.shape[0]:
                #         logger.warning(f"Cached segmentation file {osp.join(self.cached_seg_path, str(frame_idx), f'clip_{idx:04d}.jsonl')} has only {item} lines. The video clip has {clip_vid.shape[0]} frames. This indicates that the segmentation for this clip was not fully dumped. Proceeding without cache.")
                #         is_complete_dump = False
                #         break
                if is_complete_dump:
                    log_event(stage="video_segment", event="track_object_segments_in_video",
                              msg=f"obtained complete segmentations for frame_idx {frame_idx} for {num_masks} frames")
                    return cached_video_segments_index
                else:
                    # TODO: remove this later. This is for debugging purposes as I don't want to re-generate the cache right now.
                    raise ValueError("Cached segmentation files are incomplete, cannot use cache.")
                    
            else:
                # logger.warning(f"Cached segmentation file {cached_seg_path_for_frame} not found. Proceeding without cache.")
                log_event(stage="video_segment", event="track_object_segments_in_video",
                          msg=f"cached segmentation file {cached_seg_path_for_frame} not found. proceeding without cache.")

        num_clips = 0
        if overwrite_cache:
            num_clips = split_video_by_frames(
                input_path=self.video_fn,
                out_dir=self.cache_dir,
                max_frames=self.MAX_NUM_FRAMES_IN_CLIP,
                overlap=self.NUM_FRAME_OVERLAP_BETWEEN_CLIPS,
            )
        else:
            expected_num_clips = self.num_frames // self.MAX_NUM_FRAMES_IN_CLIP
            if self.num_frames % self.MAX_NUM_FRAMES_IN_CLIP != 0:
                expected_num_clips += 1
            actual_num_clips = len(os.listdir(self.cache_dir)) if osp.exists(self.cache_dir) else 0

            if actual_num_clips != expected_num_clips:
                # logger.info(f"Cache directory {self.cache_dir} has {actual_num_clips} clips, but expected {expected_num_clips}. Re-splitting the video.")
                log_event(stage="video_segment", event="track_object_segments_in_video",
                          msg=f"cache directory {self.cache_dir} has {actual_num_clips} clips, but expected {expected_num_clips}. re-splitting the video.")
                shutil.rmtree(self.cache_dir)
                num_clips = split_video_by_frames(
                    input_path=self.video_fn,
                    out_dir=self.cache_dir,
                    max_frames=self.MAX_NUM_FRAMES_IN_CLIP,
                    overlap=self.NUM_FRAME_OVERLAP_BETWEEN_CLIPS,
                )
            else:
                num_clips = actual_num_clips
        
        # load all the clip directories here
        all_clip_fns = sorted([
            osp.join(self.cache_dir, "clip_{i:04d}".format(i=i))
            for i in range(num_clips)
        ])
        
        

                
        def _dump_clip_segs(clip_seg_dict, start_frame_idx):

            """dumping clip segmentations to the
            cache dile

            Args:
                clip_seg_dict (List[Dict[str, np.ndarray]]): List of dictionaries mapping object IDs to their corresponding segmentation masks across the clip.
                start_frame_idx (int): The starting frame index of the clip in the original video.

            Raises:
                ValueError: If the cached_seg_path is None.
            """

            if self.cached_seg_path is None:
                raise ValueError("cached_seg_path is None, cannot dump clip segments.")

            prompt_dir = osp.join(self.cached_seg_path, str(frame_idx))
            if subspl_frame_idx is not None:
                prompt_dir = osp.join(self.cached_seg_path, f"raw_{frame_idx}_subspl_{subspl_frame_idx}")
            os.makedirs(prompt_dir, exist_ok=True)

            clip_idx = start_frame_idx // self.MAX_NUM_FRAMES_IN_CLIP
            clip_dump_fn = osp.join(prompt_dir, f"clip_{clip_idx:04d}.jsonl")
            
            dump_list = [{"prompt_frame_idx": int(frame_idx),
                          "frame_idx": int(start_frame_idx + f_idx),
                          "masks": {str(part_id): encode_mask(clip_seg_dict[f_idx][part_id]) 
                                    for part_id in clip_seg_dict[f_idx]},
                            "video_fn": self.video_fn # we have ended up with many variations of the video_fn so to keep track
                          } for f_idx in range(len(clip_seg_dict))]
            if subspl_frame_idx is not None:
                for item in dump_list:
                    item["prompt_frame_idx_subspl"] = int(subspl_frame_idx)
            
            with open(clip_dump_fn, "w") as f:
                f.write("\n".join([json.dumps(item) for item in dump_list]) + "\n")
            logger.info(f"Dumped clip segments to {clip_dump_fn}")

        # import ipdb; ipdb.set_trace()
        # Step 1: first generate tracks in the clip that contains frame_idx
        prompt_frame_clip_idx = frame_idx // self.MAX_NUM_FRAMES_IN_CLIP
        prompt_frame_idx_in_clip = frame_idx % self.MAX_NUM_FRAMES_IN_CLIP

        if prompt_frame_clip_idx != 0:
            prompt_frame_idx_in_clip += self.NUM_FRAME_OVERLAP_BETWEEN_CLIPS  # account for the overlap frames
        
        sgmts_in_clip_with_prompt_frame = self._track_video_object_segments_in_clip(
            video=all_clip_fns[prompt_frame_clip_idx],
            prompt_to_track=prompt_to_track,
            frame_idx=prompt_frame_idx_in_clip,
            non_overlap_masks=non_overlap_masks,
            vos_model=vos_model,
            prompt_mode=prompt_mode,
            device=device,
        )
        expected_num_frames_in_clip = self.MAX_NUM_FRAMES_IN_CLIP
        if prompt_frame_clip_idx != 0:
            expected_num_frames_in_clip += self.NUM_FRAME_OVERLAP_BETWEEN_CLIPS
        if prompt_frame_clip_idx == num_clips -1:
            expected_num_frames_in_clip = self.num_frames % self.MAX_NUM_FRAMES_IN_CLIP
            if num_clips > 1:
                expected_num_frames_in_clip += self.NUM_FRAME_OVERLAP_BETWEEN_CLIPS
    
        assert len(sgmts_in_clip_with_prompt_frame) == expected_num_frames_in_clip, \
            f"Expected {expected_num_frames_in_clip} frames in clip {prompt_frame_clip_idx}, but got {len(sgmts_in_clip_with_prompt_frame)}"

        if not incremental_dump:
            complete_sgmts = sgmts_in_clip_with_prompt_frame
        elif incremental_dump and self.cached_seg_path is not None:
            # import ipdb; ipdb.set_trace()
            frame_segs_to_dump = sgmts_in_clip_with_prompt_frame
            if prompt_frame_clip_idx != 0:
                # exclude the overlap frames  that will be included from the previous clip
                frame_segs_to_dump = sgmts_in_clip_with_prompt_frame[self.NUM_FRAME_OVERLAP_BETWEEN_CLIPS:]  
            _dump_clip_segs(
                frame_segs_to_dump,
                start_frame_idx=prompt_frame_clip_idx * self.MAX_NUM_FRAMES_IN_CLIP,
            )
            
        # Step 2: propagate the tracks in all clips AFTER the clip with the prompt frame
        prev_clip_sgmnts = sgmts_in_clip_with_prompt_frame
        for clip_idx in range(prompt_frame_clip_idx+1, num_clips):
            # break
            # TODO: just for testing. Remove later.
            if debug_stride is not None and clip_idx > prompt_frame_clip_idx + debug_stride:
                # free up memory
                break
            
            prompts_to_track_in_clip = prev_clip_sgmnts[-1]

            sgmts_in_clip = self._track_video_object_segments_in_clip(
                video=all_clip_fns[clip_idx],
                prompt_to_track=prompts_to_track_in_clip,
                frame_idx=0,
                vos_model=vos_model,
                prompt_mode=prompt_mode,
                device=device,
            )
            expected_num_frames_in_clip = self.MAX_NUM_FRAMES_IN_CLIP
            if clip_idx != 0:
                expected_num_frames_in_clip += self.NUM_FRAME_OVERLAP_BETWEEN_CLIPS
            if clip_idx == num_clips -1:
                expected_num_frames_in_clip = self.num_frames % self.MAX_NUM_FRAMES_IN_CLIP
                if num_clips > 1:
                    expected_num_frames_in_clip += self.NUM_FRAME_OVERLAP_BETWEEN_CLIPS
            assert len(sgmts_in_clip) == expected_num_frames_in_clip, \
                f"Expected {expected_num_frames_in_clip} frames in clip {clip_idx}, but got {len(sgmts_in_clip)}"

            if incremental_dump and self.cached_seg_path is not None:
                _dump_clip_segs(
                    sgmts_in_clip[self.NUM_FRAME_OVERLAP_BETWEEN_CLIPS:], # exclude the first frame as it is already included
                    start_frame_idx=clip_idx * self.MAX_NUM_FRAMES_IN_CLIP,
                )
            else:
                complete_sgmts.extend(sgmts_in_clip[self.NUM_FRAME_OVERLAP_BETWEEN_CLIPS:])
            prev_clip_sgmnts = sgmts_in_clip
                
                
        # Step 3: propagate the tracks in all clips BEFORE the clip with the prompt frame
        prev_clip_sgmnts = sgmts_in_clip_with_prompt_frame
        for clip_idx in range(prompt_frame_clip_idx-1, -1, -1):
            # break
            if debug_stride is not None and clip_idx < prompt_frame_clip_idx - debug_stride:
                # free up memory
                break
            
            # prompts_to_track_in_clip = {
            #     part_id: prev_clip_sgmnts[part_id][0]  # use the first frame's mask as the prompt
            #     for part_id in prev_clip_sgmnts
            # }
            prompts_to_track_in_clip = prev_clip_sgmnts[0]
            num_frames_in_clip = self.MAX_NUM_FRAMES_IN_CLIP 
            if clip_idx != 0:
                num_frames_in_clip += self.NUM_FRAME_OVERLAP_BETWEEN_CLIPS
            try:
                sgmts_in_clip = self._track_video_object_segments_in_clip(
                    video=all_clip_fns[clip_idx],
                    prompt_to_track=prompts_to_track_in_clip,
                    frame_idx=num_frames_in_clip-1,
                    vos_model=vos_model,
                    prompt_mode=prompt_mode,
                    device=device,
                )
                expected_num_frames_in_clip = self.MAX_NUM_FRAMES_IN_CLIP
                if clip_idx != 0:
                    expected_num_frames_in_clip += self.NUM_FRAME_OVERLAP_BETWEEN_CLIPS
                if clip_idx == num_clips -1:
                    expected_num_frames_in_clip = self.num_frames % self.MAX_NUM_FRAMES_IN_CLIP
                    if num_clips > 1:
                        expected_num_frames_in_clip += self.NUM_FRAME_OVERLAP_BETWEEN_CLIPS
                assert len(sgmts_in_clip) == expected_num_frames_in_clip, \
                    f"Expected {expected_num_frames_in_clip} frames in clip {clip_idx}, but got {len(sgmts_in_clip)}"
            except:
                import ipdb; ipdb.set_trace()
            if incremental_dump and self.cached_seg_path is not None:
                frame_segs_to_dump = sgmts_in_clip
                if clip_idx != 0:
                    frame_segs_to_dump = sgmts_in_clip[self.NUM_FRAME_OVERLAP_BETWEEN_CLIPS:]  # exclude the last frame as it is already included
                
                _dump_clip_segs(
                    frame_segs_to_dump,
                    start_frame_idx=clip_idx * self.MAX_NUM_FRAMES_IN_CLIP,
                )

            else:
                frame_segs_to_store = sgmts_in_clip
                if clip_idx != 0:
                    frame_segs_to_store = sgmts_in_clip[self.NUM_FRAME_OVERLAP_BETWEEN_CLIPS:]  # exclude the last frame as it is already included
                frame_segs_to_store.extend(complete_sgmts)
                complete_sgmts = frame_segs_to_store
            
            prev_clip_sgmnts = sgmts_in_clip
                
        if not incremental_dump:
            return complete_sgmts
    

    @api_desc(
        description="",
        export=False,
    )
    def extract_cuts_from_video(
        self,
        video_fn: str,
    ):
        """Extract cuts from video using PySceneDetect.

        Args:
            video_fn (str): Path to the video file.
        Returns:
            List[Tuple[float, float]]: List of (start_time, end_time) tuples for each cut.
        """
        pass

    def forward(
        self,
        model_name: str,
        *args,
        init_args=None,
        **kwargs
    ):
        return forward(model_name, *args, init_args=init_args, **kwargs)

    @api_desc(
        description="""
        Clears the cache directory by deleting all files in it.
        """,
        export=False
    )
    def clear_cache(self):
        """Clears the cache directory by deleting all files in it."""
        for p in os.scandir(self.cache_dir):
            if p.is_file():
                os.remove(p.path)
                
        shutil.rmtree(self.cache_dir)
    
    @staticmethod
    def _examples(version: str = "v0"):
        pass
    
       
        
                
if __name__ == "__main__":
    import json
    from pycocotools.mask import decode
    from src.tva.utils.video import find_frame_indices
    from src.annotation_tools.segmentation.preprocessing.vis_utils import overlay_mask_on_image
    
    vid_fn = osp.join(root, "data", "videos", "Chair", "mammut_1", "FoVtnbm0hPc", "FoVtnbm0hPc.mp4")
    cache_dir = osp.join(root, "tmp", "tva_test", "FoVtnbm0hPc_video_segment_test")
    metadata_fn = osp.join(root, "data", "frames-metadata", "Chair", "mammut_1", "FoVtnbm0hPc", "FoVtnbm0hPc_frames_metadata.jsonl")

    video_segmenter = VideoSegment(
        video_fn=vid_fn,
        cache_dir=cache_dir,
        multi_mask=False,
    )
    
    masks_fn = osp.join(root, "data", "segmentation-masks", "Chair",\
                    "mammut_1", "FoVtnbm0hPc", f"FoVtnbm0hPc.json")
    frame_idx_in_subspled_video = 1
    with open(masks_fn, "r") as f:
        masks_data = json.load(f)["manual"]
    masks_to_track = masks_data[str(frame_idx_in_subspled_video)]
    masks_to_track = {
        part_id: decode(rle) for part_id, rle in masks_to_track.items()
    }
    
    frame_meta = None
    all_frame_times = list()
    
    with open(metadata_fn, "r") as f:
        for line_idx, line in enumerate(f):
            line_frame_meta = json.loads(line.strip())
            if line_idx == frame_idx_in_subspled_video:
                frame_meta = line_frame_meta
                break
    
    all_frame_times = frames_between_vfr(
        vid_fn,
        start_time=0,
        end_time=None,
        return_fps=False,
        only_return_frame_timestamps=True,
    )
    frame_time = frame_meta["frame_time"]
    frame_idx_in_raw_video = find_frame_indices(all_frame_times, frame_time)[0][0]
    # import ipdb; ipdb.set_trace()
    debug_stride = 1
    complete_sgmts = video_segmenter.track_object_segments_in_video(
        prompt_to_track=masks_to_track,
        frame_idx=frame_idx_in_raw_video,
        vos_model="sam2",
        prompt_mode="mask",
        device="cuda",
        debug_stride=debug_stride,
    )
    
    # import ipdb; ipdb.set_trace()
    clip_fns = sorted(
        [osp.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith(".mp4") and f.startswith("clip_")],
        key=lambda x: int(x.split("clip_")[-1].split(".mp4")[0])
    )  
    clip_idx_with_prompt_frame = frame_idx_in_raw_video // video_segmenter.MAX_NUM_FRAMES_IN_CLIP
    clip_fns = clip_fns[clip_idx_with_prompt_frame-debug_stride: clip_idx_with_prompt_frame+debug_stride+1]
    raw_video_frames = np.concatenate([media.read_video(clip_fn) for clip_fn in clip_fns], axis=0)
    new_video_frames = raw_video_frames.copy().astype(float)
    # raw_video_frames = media.read_video(vid_fn)
    num_frames_with_sgmts = min(len(list(complete_sgmts.values())[0]),
                                len(raw_video_frames))
    for frame_idx in range(num_frames_with_sgmts):
        frame = raw_video_frames[frame_idx]
        segmt = {
            part_id: complete_sgmts[part_id][frame_idx] 
            for part_id in complete_sgmts
        }

        overlaid = frame.astype(float).copy() / 255.0
        # import ipdb; ipdb.set_trace()
        for part_id in segmt:
            # import ipdb; ipdb.set_trace()
            overlaid[segmt[part_id]>0] = overlay_mask_on_image(
                overlaid, 
                segmt[part_id], 
                obj_id=int(part_id), 
                alpha=0.5
            )[segmt[part_id]>0]
        # import ipdb; ipdb.set_trace() 
        new_video_frames[frame_idx] = overlaid
    # import ipdb; ipdb.set_trace()
    media.write_video(osp.join(cache_dir, f"tracked_FoVtnbm0hPc_full.mp4"), new_video_frames, fps=20)
    print("Done")
