from typing import Literal, Optional, List, Union, Any

import re
import os
# # if using Apple MPS, fall back to CPU for unsupported ops
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import tempfile
import shutil
import os.path as osp

from typing import Dict, List, Literal, Union
import numpy as np
import torch
if torch.cuda.is_available():
    torch.autocast("cuda", dtype=torch.float32).__enter__()
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import mediapy as media
import cv2
import torch
from sam2.build_sam import build_sam2_video_predictor


def _natural_key(name: str):
    """Key for human-friendly sorting by embedded numbers."""
    parts = re.split(r'(\d+)', name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]

def _compute_indices(total, n=None, start=0, step=1, stop=None, indices=None):
    """
    Normalize frame selection to a sorted, unique list of indices within [0, total).
    Priority: explicit indices > (start, step, n/stop)
    """
    if indices is not None:
        idx = sorted(set(i for i in indices if 0 <= i < total))
        return idx

    if stop is None:
        if n is None:
            raise ValueError("Provide either indices or (start, step, n) or (start, step, stop).")
        stop = min(total, start + step * n)
    else:
        stop = min(total, stop)

    if step <= 0:
        raise ValueError("step must be a positive integer")

    return list(range(start, stop, step))

def get_frame_subsequence(
    input_path: str,
    *,
    # Choose one of:
    indices=None,      # e.g., [0, 2, 5, 9]
    start: int = 0,    # used if indices is None
    step: int = 1,     # used if indices is None
    n: int | None = None,  # used if indices is None and stop is None
    stop: int | None = None,  # alternative to n
    fps: int = 1
):
    """
    Extract a subsequence of frames from either a video (.mp4) or a directory of images.

    Args:
        input_path: Path to an .mp4 video file or a directory containing images.
        indices: Explicit list of frame indices to select (highest priority).
        start: Start index if not using 'indices'.
        step: Step size for the subsequence.
        n: Number of frames to take (with start/step). Mutually exclusive with 'stop'.
        stop: End index (exclusive) for the subsequence. Mutually exclusive with 'n'.
        fps: FPS for the output video when input is .mp4.

    Returns:
        str | list[str]:
            - If input is .mp4: path to a temporary .mp4 containing only the selected frames.
            - If input is a directory: list of file paths (copied to a temp dir) of selected images.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} does not exist")

    # Video case
    if os.path.isfile(input_path) and input_path.lower().endswith(".mp4"):
        # Read full video into frames (array-like [T, H, W, 3])
        vid = media.read_video(input_path)
        total = len(vid)
        sel = _compute_indices(total, n=n, start=start, step=step, stop=stop, indices=indices)
        if not sel:
            raise ValueError("No frames selected (empty subsequence).")

        # Gather selected frames
        subframes = [vid[i] for i in sel]

        # Write to a temporary mp4
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_path = tmp.name
        tmp.close()
        media.write_video(tmp_path, subframes, fps=fps)
        return tmp_path

    # Directory of images case
    elif os.path.isdir(input_path):
        # Sort images naturally (by embedded numbers if present)
        all_imgs = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        ]
        all_imgs.sort(key=lambda p: _natural_key(os.path.basename(p)))

        total = len(all_imgs)
        if total == 0:
            raise ValueError("No images found in the directory.")

        sel = _compute_indices(total, n=n, start=start, step=step, stop=stop, indices=indices)
        if not sel:
            raise ValueError("No images selected (empty subsequence).")

        temp_dir = tempfile.mkdtemp()
        copied = []
        for i, idx in enumerate(sel):
            src = all_imgs[idx]
            stem, ext = os.path.splitext(os.path.basename(src))
            dst = os.path.join(temp_dir, f"frame_{i:04d}{ext}")
            shutil.copy(src, dst)
            copied.append(dst)
        return copied

    else:
        raise ValueError("Input path must be an .mp4 file or a directory of images")


class SAM2:
    def __init__(self, 
                 sam2_checkpoint = "ckpts/sam2.1_hiera_large.pt", 
                 model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml", 
                 device = "cuda",
                 multi_mask=False,
                 non_overlap_masks=False):
        
        """
        Some key notes to remember for object IDs vs. object indices in SAM:
        - in our case, while adding masks, we type cast object IDs to int 
        - but when collating outputs, we still use object IDs as str (to be consistent with user input)
        - object indices are always int and are used internally by the model
        - when accessing object IDs in the inference state, type case object IDs to int, as the model is told that these are int
        """
        
        hydra_overrides = []
        if non_overlap_masks:
            hydra_overrides.append("++model.non_overlap_masks=true")
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, 
                                                    device=device, hydra_overrides_extra=hydra_overrides)

        implementation_dir = {
            "sam2": "objtrack/sam2",
            "sam2.1": "objtrack/sam2",
            "samurai": "objtrack/samurai"
        }[model_cfg.split('/')[1]]
    
        self.multi_mask = multi_mask

    def initialize(self, **info):
        offload_video_to_cpu = info.get('offload_video_to_cpu', False)
        self.inference_state = self.predictor.init_state(video_path=info['video_dir'], 
                                                         offload_video_to_cpu=offload_video_to_cpu,
                                                         async_loading_frames=True)
    
    def track(self, mask, frame_idx=0):
        self.predictor.reset_state(self.inference_state)
        
        _, _, _ = self.predictor.add_new_mask(
            inference_state=self.inference_state,
            frame_idx=frame_idx,
            obj_id=0,
            mask=mask,
        )
        
        # output
        # |-> "prediction"
        #     |-> frame_idx
        #         |-> obj_id: np.ndarray
        # |-> "obj_score"
        #     |-> frame_idx
        #         |-> obj_id: float
        # |-> "multi_masks"
        #     |-> frame_idx
        #         |-> obj_id
        #             |-> rank: np.ndarray
        # |-> "multi_masks_pred_ious"
        #     |-> frame_idx
        #         |-> obj_id
        #             |-> rank: float
        output = {'prediction': dict(), 'obj_score': dict()}
        if self.multi_mask:
            output['multi_masks'] = dict()
            output['multi_masks_pred_ious'] = dict()

        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            output['prediction'][out_frame_idx] = {
                obj_idx: (out_mask_logits[i, 0] > 0.0).cpu().numpy() 
                    for i, obj_idx in enumerate(out_obj_ids)
            }
        
            if out_frame_idx in self.inference_state['output_dict_per_obj'][out_obj_ids[0]]['non_cond_frame_outputs']:
                frame_out = self.inference_state['output_dict_per_obj'][out_obj_ids[0]]['non_cond_frame_outputs'][out_frame_idx]

                output['obj_score'][out_frame_idx] = {
                    obj_idx: frame_out['object_score_logits'][i,0].item() 
                        for i, obj_idx in enumerate(out_obj_ids)
                }

                # NOTE (ac): at present we don't care about `multi_masks` and `multi_masks_pred_ious`
                if self.multi_mask:
                    _, multimasks = self.predictor._get_orig_video_res_output(self.inference_state, frame_out['low_res_multimasks'])

                    for i, obj_idx in enumerate(out_obj_ids):
                        output['multi_masks'][out_frame_idx] = {obj_idx: dict()}
                        output['multi_masks_pred_ious'][out_frame_idx] = {obj_idx: dict()}

                        ious_rank = np.argsort(frame_out['ious'][i].cpu().numpy())[::-1]
                        output['multi_masks'][out_frame_idx][obj_idx] = {
                            ii: (multimasks[i, rank] > 0.0).cpu().numpy() 
                                for ii, rank in enumerate(ious_rank)
                        }

                        output['multi_masks_pred_ious'][out_frame_idx][obj_idx] = {
                            ii: frame_out['ious'][i, rank].item() 
                                for ii, rank in enumerate(ious_rank)
                        }
        return output

    def add_mask_prompts(self, masks, frame_indices=[0], obj_ids=[0]):
        
        self.predictor.reset_state(self.inference_state)
        for i, (mask, frame_idx, obj_id) in enumerate(zip(masks, frame_indices, obj_ids)):
            _, _, _ = self.predictor.add_new_mask(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=int(obj_id),
                mask=mask,
            )
            
    def add_point_prompts(self, points, frame_indices=[0], obj_ids=[0]):
        self.predictor.reset_state(self.inference_state)
        for i, (frame_obj_pts, frame_idx, obj_id) in enumerate(zip(points, frame_indices, obj_ids)):
            _, _, _ = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=int(obj_id),
                points=frame_obj_pts,
                labels=np.ones((frame_obj_pts.shape[0],), dtype=np.int32), # positive click
            )
    
    
    def track_multiple_objects_in_video(
        self, 
        prompts, 
        mode="masks", 
        frame_indices=[0], 
        obj_ids=['0'],
        track_points_by_first_appearance=False,
        full_sweep=True
    ):
        
        """Tracks multiple objects in a video given initial prompts.
        
        For frames after the frame index of the prompt, we track in forward mode.
        For frames before the frame index of the prompt, we track in backward mode.
        (Even though the propagation is done on the full video in the forward and backward directions,
        the masks are only added at the specified frame indices.)
        """
        
        
        # get first appearance of each unique object in obj_ids from frame_indices
        obj_ids_first_appearance = dict()
        for i, frame_idx in enumerate(frame_indices):
            obj_id = obj_ids[i]
            if obj_id not in obj_ids_first_appearance:
                obj_ids_first_appearance[obj_id] = frame_idx
            else:
                # should be redundant but just in case
                obj_ids_first_appearance[obj_id] = min(obj_ids_first_appearance[obj_id], frame_idx)

        start_frame_index = None
        if len(list(set(list(obj_ids_first_appearance.values())))) == 1:
            start_frame_index = list(obj_ids_first_appearance.values())[0]
        if full_sweep:
            start_frame_index = None
        
        if mode == "masks":
            self.add_mask_prompts(prompts, frame_indices, obj_ids)
        elif mode == "points":
            # print("adding prompts")
            if track_points_by_first_appearance:
                # add points only for the first appearance of each object
                new_prompts = [prompts[pidx] for obj_id, first_appearance in obj_ids_first_appearance.items() for pidx in range(len(prompts)) if frame_indices[pidx] == first_appearance and obj_ids[pidx] == obj_id]
                new_frame_indices = [frame_indices[pidx] for obj_id, first_appearance in obj_ids_first_appearance.items()  for pidx in range(len(prompts)) if frame_indices[pidx] == first_appearance and obj_ids[pidx] == obj_id]
                new_obj_ids = [obj_ids[pidx] for obj_id, first_appearance in obj_ids_first_appearance.items()  for pidx in range(len(prompts)) if frame_indices[pidx] == first_appearance and obj_ids[pidx] == obj_id]
                prompts = new_prompts
                frame_indices = new_frame_indices
                obj_ids = new_obj_ids
                # print(len(prompts), len(frame_indices), len(obj_ids))
                assert len(prompts) == len(obj_ids_first_appearance)
            self.add_point_prompts(prompts, frame_indices, obj_ids)

        output = {'prediction': dict(), 'obj_score': dict()}
        if self.multi_mask:
            output['multi_masks'] = dict()
            output['multi_masks_pred_ious'] = dict()
        
        def _collate_output_from_frame_idx_for_obj_id(
            out_frame_idx: int,
            out_obj_idx: int, 
            out_obj_id: str, 
            out_mask_logits: torch.Tensor, 
            output_dict: Dict = None, 
            multimasks: Any = None,
        ):
            """Collate the output of a specific frame index and object ID into the output dictionary.
            using the output logits and the inference state.

            Args:
                out_frame_idx (int): frame index in the video
                out_obj_idx (int): object ID as an integer (that is how the model stores our object ID)
                out_obj_id (str): object ID in the video (used to store the output)
                out_mask_logits (torch.Tensor): mask logits for the object
                output_dict (Dict, optional): output dictionary to collate the results. Defaults to None.
                multimasks (Any, optional): cached multi masks for the objects in the video. Defaults to None.
            Returns: 
                output_dict (Dict): updated output dictionary with the collated results.
                    Structure:
                        |-> "prediction"
                            |-> frame_idx
                                |-> obj_id: np.ndarray
                        |-> "obj_score"
                            |-> frame_idx
                                |-> obj_id: float
                        |-> "multi_masks"
                            |-> frame_idx
                                |-> obj_id
                                    |-> rank: np.ndarray
                        |-> "multi_masks_pred_ious"
                            |-> frame_idx
                                |-> obj_id
                                    |-> rank: float
            """
            
            obj_id = out_obj_id
            if output_dict is None:
                output_dict = {"prediction": dict(), "obj_score": dict()}
                if self.multi_mask:
                    if 'multi_masks' not in output_dict:
                        output_dict['multi_masks'] = dict()
                    if 'multi_masks_pred_ious' not in output_dict:
                        output_dict['multi_masks_pred_ious'] = dict()
                
            if out_frame_idx not in output_dict["prediction"]:
                output_dict["prediction"][out_frame_idx] = dict()

            # setting the predicted mask
            output_dict["prediction"][out_frame_idx][obj_id] = (out_mask_logits[0, 0] > 0.0).cpu().numpy()
            
            if out_frame_idx in self.inference_state['output_dict_per_obj'][out_obj_idx]['non_cond_frame_outputs']:
                frame_out = self.inference_state['output_dict_per_obj'][out_obj_idx]['non_cond_frame_outputs'][out_frame_idx]

                if out_frame_idx not in output_dict["obj_score"]:
                    output_dict["obj_score"][out_frame_idx] = dict()
                
                # setting the object score
                output_dict["obj_score"][out_frame_idx][obj_id] = frame_out['object_score_logits'][out_obj_idx,0].item()

                if self.multi_mask:
                    # _, multimasks = self.predictor._get_orig_video_res_output(self.inference_state, frame_out['low_res_multimasks'])
                    if out_frame_idx not in output_dict["multi_masks"]:
                        output_dict["multi_masks"][out_frame_idx] = dict()
                    if out_frame_idx not in output_dict["multi_masks_pred_ious"]:
                        output_dict["multi_masks_pred_ious"][out_frame_idx] = dict()
                    
                    if obj_id not in output_dict["multi_masks"][out_frame_idx]:
                        output_dict["multi_masks"][out_frame_idx][obj_id] = dict()
                    if obj_id not in output_dict["multi_masks_pred_ious"][out_frame_idx]:
                        output_dict["multi_masks_pred_ious"][out_frame_idx][obj_id] = dict()
                    
                    
                    ious_rank = np.argsort(frame_out['ious'][out_obj_idx].cpu().numpy())[::-1]
                    output['multi_masks'][out_frame_idx][obj_id] = {
                        ii: (multimasks[out_obj_idx, rank] > 0.0).cpu().numpy() 
                            for ii, rank in enumerate(ious_rank)
                    }

                    output['multi_masks_pred_ious'][out_frame_idx][obj_id] = {
                        ii: frame_out['ious'][out_obj_idx, rank].item() 
                            for ii, rank in enumerate(ious_rank)
                    }
                    
            return output_dict

        output = None
        for out_frame_idx, out_obj_indices, out_mask_logits in self.predictor.propagate_in_video(
            self.inference_state, reverse=False, 
            start_frame_idx=start_frame_index if start_frame_index is not None else 0,
        ):
            
            for i, out_obj_idx in enumerate(out_obj_indices):
                
                if not self.multi_mask:
                    output = {"prediction": dict(), "obj_score": dict()} if output is None else output
                    if out_frame_idx not in output["prediction"]:
                        output["prediction"][out_frame_idx] = dict()
                    output["prediction"][out_frame_idx][str(out_obj_idx)] = (out_mask_logits[i, 0] > 0.0).cpu().numpy()
                else:
                    multimasks = None
                    # import ipdb; ipdb.set_trace()
                    try:
                        if out_frame_idx in self.inference_state['output_dict_per_obj'][out_obj_idx]['non_cond_frame_outputs']:
                            frame_out = self.inference_state['output_dict_per_obj'][out_obj_idx]['non_cond_frame_outputs'][out_frame_idx]
                            _, multimasks = self.predictor._get_orig_video_res_output(
                                self.inference_state, frame_out['low_res_multimasks'])
                        output = _collate_output_from_frame_idx_for_obj_id(
                            out_frame_idx=out_frame_idx,
                            out_obj_id=str(out_obj_idx),
                            out_obj_idx=out_obj_idx,
                            out_mask_logits=out_mask_logits[i:i+1],
                            output_dict=output,
                            multimasks=multimasks,
                        )
                    except:
                        import ipdb; ipdb.set_trace()
                    
        num_frames = self.inference_state["num_frames"]
        for out_frame_idx, out_obj_indices, out_mask_logits in self.predictor.propagate_in_video(
            self.inference_state, reverse=True, 
            start_frame_idx=start_frame_index if start_frame_index is not None else num_frames-1,
        ):
           
            for i, out_obj_idx in enumerate(out_obj_indices):
                if out_frame_idx <= obj_ids_first_appearance[str(out_obj_idx)]:
                    if not self.multi_mask:
                        if out_frame_idx not in output["prediction"]:
                            output["prediction"][out_frame_idx] = dict()
                        out_mask = (out_mask_logits[i, 0] > 0.0).cpu().numpy().astype(np.uint8)
                        output["prediction"][out_frame_idx][str(out_obj_idx)] = out_mask
                    else:
                        multimasks = None
                        if out_frame_idx in self.inference_state['output_dict_per_obj'][out_obj_idx]['non_cond_frame_outputs']:
                            frame_out = self.inference_state['output_dict_per_obj'][out_obj_idx]['non_cond_frame_outputs'][out_frame_idx]
                            _, multimasks = self.predictor._get_orig_video_res_output(
                                self.inference_state, frame_out['low_res_multimasks'])
                        output = _collate_output_from_frame_idx_for_obj_id(
                            out_frame_idx=out_frame_idx,
                            out_obj_id=str(out_obj_idx),
                            out_obj_idx=i,
                            out_mask_logits=out_mask_logits[i:i+1],
                            output_dict=output,
                            multimasks=multimasks,
                        )
        
        return output

    
    def clear_all_cache(self):
        for k in self.inference_state.keys():
            self.inference_state[k] = None
        torch.cuda.empty_cache()


class SAM2Wrapper(SAM2):
    
    def __init__(
        self,
        sam2_checkpoint = osp.join(osp.dirname(osp.abspath(__file__)), "ckpts/sam2.1_hiera_large.pt"), 
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml", 
        device = "cuda",
        multi_mask=False,
        non_overlap_masks=False,
    ):
        super().__init__(
            sam2_checkpoint=sam2_checkpoint, model_cfg=model_cfg, 
            device=device, multi_mask=multi_mask, non_overlap_masks=non_overlap_masks
        )

    def forward(
        self,
        video_dir: Union[str, List[str]],
        masks: List[np.ndarray],
        frame_idxs: List[int]=[0],
        obj_ids: List[str]=['0'],
        mode: Literal["masks", "points"] = "masks",
        offload_video_to_cpu: bool=False,
        track_points_by_first_appearance: bool=False,
        full_sweep: bool=True,
        **kwargs
    ):
        """
        Args:
            video_dir (str): path to the video directory or video file
            masks (List[np.ndarray]): list of binary masks for each object
            frame_idxs (List[int], optional): List of frame indices to track.
            obj_ids (List[str], optional): List of object IDs corresponding to each mask. Defaults to ['0'].
            mode (Literal["masks", "points"], optional): type of prompts provided. Defaults to "masks".
            offload_video_to_cpu (bool, optional): whether to offload video frames to CPU. Defaults to False.
            track_points_by_first_appearance (bool, optional): if mode is "points", whether to track points by their first appearance. Defaults to False.
            full_sweep (bool, optional): whether to perform full sweep tracking. Defaults to False.
        """
        
        self.initialize(video_dir=video_dir, offload_video_to_cpu=offload_video_to_cpu)
        output = self.track_multiple_objects_in_video(
            prompts=masks,
            frame_indices=frame_idxs,
            obj_ids=obj_ids,
            mode=mode,
            track_points_by_first_appearance=True if \
                track_points_by_first_appearance and mode=="points" else False,
            full_sweep=full_sweep,
        )
 
        return output

    def clear_all_cache(self):
        return super().clear_all_cache()
    


class SplitSAM2(SAM2):

    def __init__(self, 
                 sam2_checkpoint = "ckpts/sam2.1_hiera_large.pt", 
                 model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml", 
                 device = "cuda",
                 multi_mask=False,
                 split_thrd=0.3):
        super().__init__(sam2_checkpoint, model_cfg, device, multi_mask)
        self.split_thrd = split_thrd
        raise Exception("Not implemented yet")

    def propagate(self, output):
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            output['prediction'][out_frame_idx] = {
                obj_idx: (out_mask_logits[i, 0] > 0.0).cpu().numpy() 
                    for i, obj_idx in enumerate(out_obj_ids)
            }

            output['split_masks'][out_frame_idx] = {obj_idx: dict() for obj_idx in out_obj_ids}
            for i, obj_idx in enumerate(out_obj_ids):
                num_labels, labels = cv2.connectedComponents(output['prediction'][out_frame_idx][obj_idx].astype(np.uint8))
                num_components = num_labels - 1
                if num_components > 1:
                    num_pix_component = np.array([np.sum(labels == i) for i in range(1, num_labels)])
                    prop = num_pix_component / np.sum(num_pix_component)
                    prop = prop[np.argsort(prop)[::-1]]
                    for p in prop[1:]:
                        if p > self.split_thrd:
                            breakpoint()

        return output, None, None
        
    def track(self, mask, frame_idx=0):
        output = {'prediction': dict(), 'split_masks': dict()}

        best_masks, best_mask_frame_idx = [mask], frame_idx
        while best_masks is not None:
            self.predictor.reset_state(self.inference_state)
            for i, best_mask in enumerate(best_masks):
                _, _, _ = self.predictor.add_new_mask(
                    inference_state=self.inference_state,
                    frame_idx=best_mask_frame_idx,
                    obj_id=i,
                    mask=best_mask,
                )
            output, best_masks, best_mask_frame_idx = self.propagate(output)
    
        return output
