import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from typing import Optional, Tuple, List
from pathlib import Path
import os
import os.path as osp
import sys
import json
import time
from fractions import Fraction

from loguru import logger

def ts_ms() -> int:
    return int(time.time() * 1000)

def log_event(
    *,
    stage: str,                 # who is logging: "agent", "video_segment", "image_patch", ...
    event: str,                 # what event: "tracking", "connectivity", ...
    msg: str = None,            # human-readable line
    file_path: str = None,      # when relevant
    mime: str = None,           # "image/png", "video/mp4", "application/json", ...
    meta: dict = None,          # any extras (prompt tokens, latency_ms, model, args, etc.)
):
    # Keep it flat + obvious. Everything ends up in the JSON record.
    record = {
        "t_ms": ts_ms(),
        "stage": stage,
        "event": event,
        "msg": msg,
        "file_path": file_path,
        "mime": mime,
        "meta": meta or {},
    }
    logger.bind(**record).info(f"{stage}.{event}: {msg or ''}")

def logging_setup(run_path: Optional[str] = None):


    LOG_PATH = Path(run_path)
    logger.remove()
    logger.add(
        LOG_PATH,
        serialize=True,  # emits structured JSON per line
        enqueue=True,
        backtrace=True,
        diagnose=True,
        level="INFO",
    )
    logger.add(sys.stderr, level="INFO")




def _fraction_to_float(frac):
    if frac is None:
        return None
    if isinstance(frac, Fraction):
        return float(frac)
    try:
        # Handles "num/den" strings too
        num, den = map(int, str(frac).replace(":", "/").split("/"))
        return num / den if den else None
    except Exception:
        return None

def dump_frames_metadata_to_json():
    import sys
    sys.path.append(osp.join(root, "src"))
    from IKEAVideo.dataloader.dataset_keyframe_fast import KeyframeVideoDataset, colors

    dump_dir = osp.join(root, "data", "frames-metadata")
    
    data_dir = osp.join(root, "data")
    annotation_file = osp.join(data_dir, "data.json")
    video_dir = osp.join(data_dir, "videos")
    obj_dir = osp.join(data_dir, 'parts')
    manual_img_dir = osp.join(data_dir, 'manual_img')
    pdf_dir = osp.join(data_dir, 'pdfs')
    frame_dir = osp.join(data_dir, "rgb-frames")

    num_of_data = None # None means all samples (videos) in the dataset are loaded, change to a number for specific number of samples
    debug = False 
    verbose = False # Set to True to print out the data
    load_into_mem=False # Load objects into memory. Will slow down instantiation of dataset object. Usually never needed
    demo_print = False 
    demo_viz = False # Set to True to visualize the data
    skip_vid_img = True # Skip loading video image. Setting to False slows index access of each object
    skip_manual_img = True # Loads IKEA manual image for the frame. Setting to False slows index access of each object

    dataset = KeyframeVideoDataset(
        annotation_file, 
        video_dir, 
        transform=None, 
        load_into_mem=load_into_mem, 
        verbose=verbose, 
        debug=debug, 
        obj_dir=obj_dir, 
        num_of_data=num_of_data, 
        manual_img_dir=manual_img_dir, 
        pdf_dir=pdf_dir, 
        demo_print=demo_print, 
        demo_viz=demo_viz, 
        skip_vid_img=skip_vid_img, 
        skip_manual_img=skip_manual_img
    )
    
    for i in range(len(dataset)):
        try:
            frames_metadata = dataset.__getitem__(i, start_idx=0, end_idx=None)
        except Exception as e:
            print(f"Error in processing video {i}, skipping...")
            continue
        furniture_type = frames_metadata[0]["category"]
        furniture_name = frames_metadata[0]["name"]
        video_id = frames_metadata[0]["video_id"].split("?v=")[-1]
        os.makedirs(osp.join(dump_dir, furniture_type, furniture_name, video_id), exist_ok=True)
        with open(osp.join(dump_dir, furniture_type, furniture_name, video_id, f"{video_id}_frames_metadata.jsonl"), "a+") as f:
            for frame_meta in frames_metadata:
                frame_meta.pop("fine_grained_meshes")
                frame_meta.pop("manual_meshes")
                frame_meta.pop("meshes")
                frame_meta.pop("mask")
                frame_meta.pop("obj_path")
                if "mask" in frame_meta["manual"]:
                    frame_meta["manual"].pop("mask")
                f.write(f"{json.dumps(frame_meta)}\n")
                
if __name__ == "__main__":
    dump_frames_metadata_to_json()
