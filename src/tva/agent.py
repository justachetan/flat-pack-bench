import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from typing import Literal, Optional, Tuple, Union, Dict, List, Any, Set, Sequence


import os
import os.path as osp
import sys
import copy
import json
import time
import glob
import uuid
import yaml
import shutil
import typing
import builtins
import functools
import traceback
import subprocess

import PIL

import numpy as np
import pandas as pd
import hydra
from pathlib import Path
from omegaconf import DictConfig
from pycocotools import mask as mask_utils
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger

from PIL import Image

from pycocotools.mask import decode

from src.eval.prompts.templates.questions.convert_yaml_to_json import convert_yaml_to_json
from src.eval.models.base_model import BaseModel
from src.tva.utils.video import convert_frame_idx_subspl_to_raw
from src.tva.media.image_patch import ImagePatch
from src.tva.media.video_segment import VideoSegment
from src.tva.utils.parser import rewrite_constructors_to_partials
from src.tva.utils.common import logging_setup, log_event

class TVAAgent:
    
    def __init__(
        self, 
        orchestrator: BaseModel,
        video_dir: str,
        frame_dir: str,
        subspl_video_dir: str,
        cached_masks_dir: str,
        prompt_masks_dir: str,
        metadata_dir: str,
        cache_dir: str,
        change_sys_prompt_for_orchestrator: bool = False,
        subspl_factor_for_videos: int = 2,
    ):
        """_summary_

        Args:
            orchestrator (BaseModel): Orchstrator LLM model instance.
            video_dir (str): Directory containing video files.
            frame_dir (str): Directory containing frame images.
            subspl_video_dir (str): Directory containing video subtitles.
            cached_masks_dir (str): Directory containing cached mask files, i.e., masks over the full video.
            prompt_masks_dir (str): Directory containing prompt mask files, i.e., masks for the visual prompts.
            metadata_dir (str): Directory containing metadata files.
            cache_dir (str): Directory for caching intermediate results.
            change_sys_prompt_for_orchestrator (bool, optional): Whether to change the system prompt for the orchestrator. Decides
                whether `task_instruction` will remain with the normal question content
                or as a standalone message. For Gemini/ChatGPT, this should be True (as they
                tend to allow changing the system prompt in their API). For OSS models 
                this choice matters and is usually False. Defaults to False.
            subspl_factor_for_videos (int, optional): Subsampling factor for the subsampled videos
        """
        
        self.orchestrator = orchestrator
        # with open(self.question_fn, 'r') as f:
        #     self.question_json = yaml.safe_load(f)

        self.video_dir = video_dir
        self.frame_dir = frame_dir
        self.subspl_video_dir = subspl_video_dir
        self.cached_masks_dir = cached_masks_dir
        self.prompt_masks_dir = prompt_masks_dir
        self.metadata_dir = metadata_dir
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.subspl_factor_for_videos = subspl_factor_for_videos
        if self.subspl_factor_for_videos < 1:
            logger.warning(
                f"Subsampling factor for videos ({self.subspl_factor_for_videos}) must be at least 1."
            )
            raise ValueError("Invalid subsampling factor configuration.")
        self.change_sys_prompt = change_sys_prompt_for_orchestrator
        log_event(stage="agent", event="init", msg=f"TVAAgent initialized with change_sys_prompt={self.change_sys_prompt}",
                  meta={
                      "video_dir": video_dir,
                      "frame_dir": frame_dir,
                      "subspl_video_dir": subspl_video_dir,
                      "cached_masks_dir": cached_masks_dir,
                      "prompt_masks_dir": prompt_masks_dir,
                      "metadata_dir": metadata_dir,
                      "cache_dir": cache_dir,
                      "subspl_factor_for_videos": subspl_factor_for_videos,
                      "change_sys_prompt_for_orchestrator": change_sys_prompt_for_orchestrator,
                  })
              
    
    
    def _load_and_verify_data(self, question_fn: Optional[str] = None, question_json: Optional[Dict] = None):

        if question_json is not None:
            log_event(stage="agent", event="_load_and_verify_data", 
                      msg="loading question payload from provided question JSON")
            self.question_json = question_json
        elif question_fn is not None:
            # logger.info(f"Loading question payload from {question_fn}")
            log_event(stage="agent", event="_load_and_verify_data", msg=f"loading question payload from {question_fn}")
            with open(question_fn, 'r') as f:
                self.question_json = yaml.safe_load(f)
        else:
            raise ValueError("Either question_fn or question_json must be provided.")

        self.furniture_category = self.question_json["category"]
        self.furniture_name = self.question_json["name"]
        self.video_id = self.question_json["video_id"]
        self.question_category = self.question_json["question_category"]
        self.keyframe_idxs_for_question = self.question_json["frame_idx"]

        if isinstance(self.keyframe_idxs_for_question, int):
            self.keyframe_idxs_for_question = [self.keyframe_idxs_for_question]
        # logger.info(
        #     f"Resolved {len(self.keyframe_idxs_for_question)} keyframe indices for video {self.question_json['video_id']}",
        # )
        log_event(stage="agent", event="_load_and_verify_data", 
                  msg=f"resolved {len(self.keyframe_idxs_for_question)} keyframe indices for video {self.question_json['video_id']}",
                  meta={
                      "keyframe_idxs_for_question": self.keyframe_idxs_for_question,
                  })
        
        mid_path = osp.join(self.furniture_category, self.furniture_name, self.video_id)
        self.video_fn = osp.join(self.video_dir,
                                 mid_path,
                                 f"{self.video_id}.mp4")
        


        self.subspl_video_fn = osp.join(self.subspl_video_dir,
                                        mid_path,
                                        f"{self.video_id}.mp4")
        if not osp.exists(self.subspl_video_fn):
            # logger.warning(
            #     f"Subsampled video file does not exist: {self.subspl_video_fn}. Looking for frame JPEGs instead."
            # )
            log_event(stage="agent", event="_load_and_verify_data", 
                      msg=f"subsampled video file does not exist: {self.subspl_video_fn}. Looking for frame JPEGs instead.")
            frame_jpegs = glob.glob(osp.join(self.subspl_video_dir, mid_path, "frames", "*.jpg"))
            if len(frame_jpegs) == 0:
                raise FileNotFoundError(f"No subsampled video or frame JPEGs found for {self.subspl_video_fn}")
            frame_jpegs = sorted(frame_jpegs, key=lambda x: int(osp.basename(x).split(".")[0]))
            # logger.info(
            #     f"Found {len(frame_jpegs)} subsampled frame JPEGs for video {self.video_id}",
            # )
            log_event(stage="agent", event="_load_and_verify_data",
                        msg=f"found {len(frame_jpegs)} subsampled frame JPEGs for video {self.video_id}",
                        meta={
                            "num_subspl_frame_jpegs": len(frame_jpegs),
                        })
            self.subspl_video_fn = osp.join(self.subspl_video_dir, mid_path, "frames")
            # logger.info(
            #     f"Using subsampled video frame directory: {self.subspl_video_fn}"
            # )
            log_event(stage="agent", event="_load_and_verify_data",
                        msg=f"using subsampled video frame directory: {self.subspl_video_fn}")

        all_timestamps = list()     
        self.metadata = list()       
        self.metadata_fn = osp.join(self.metadata_dir, mid_path, f"{self.video_id}_frames_metadata.jsonl")
        with open(self.metadata_fn, 'r') as f:
            for line_idx, line in enumerate(f):
                line_metadata = json.loads(line)
                if line_idx in self.keyframe_idxs_for_question:
                    self.metadata.append(line_metadata)
                all_timestamps.append(line_metadata['frame_time'])
        # logger.info(
        #     f"Collected metadata for {len(self.metadata)} frames of video {self.video_id}",
        # )
        log_event(stage="_load_and_verify_data", event="load_metadata", 
                  msg=f"collected metadata for {len(self.metadata)} frames of video {self.video_id}",
                  meta={
                      "num_metadata_frames": len(self.metadata),
                  })
        
        all_timestamps = np.array(all_timestamps)
        self.keyframe_times_in_raw_video = [i["frame_time"] for i in self.metadata]


        if osp.isdir(self.subspl_video_fn):
            subspl_vid_metadata_fn = osp.join(osp.dirname(self.subspl_video_fn), "subspl_frames_metadata.csv")
            metadata_df = pd.read_csv(subspl_vid_metadata_fn)

            if self.subspl_factor_for_videos > 1:
                # adjust for subsampling factor of agent
                df = metadata_df.copy()
                df_filtered = pd.concat(
                    [df.iloc[::self.subspl_factor_for_videos], df[df["is_prompt_frame"]]],
                    ignore_index=False
                )
                df_filtered = (
                    df_filtered
                    .drop_duplicates(subset=["subsampled_frame_idx"])
                    .sort_values("subsampled_frame_idx")
                    .reset_index(drop=True)
                )
                df_filtered["subsampled_frame_idx"] = df_filtered.index
                metadata_df = df_filtered
                
            result_df = metadata_df[metadata_df["is_prompt_frame"]][["subsampled_frame_idx", "keyframe_idx"]]
            mapping = result_df.set_index("keyframe_idx")["subsampled_frame_idx"].to_dict()
            # segmentation cache file is named using this index
            self.keyframe_idxs_in_subspl_video = [mapping[item] for item in self.keyframe_idxs_for_question]
            # import ipdb; ipdb.set_trace()

        else:
            # NOTE: is legacy and not tested, likely will not be used either as we moved to frame dirs for subspl videos
            self.keyframe_idxs_in_subspl_video = convert_frame_idx_subspl_to_raw(
                subspl_frame_idx=self.keyframe_idxs_for_question,
                vid_fn=self.subspl_video_fn,
                all_frame_timestamps=all_timestamps,
                metadata_fn=self.metadata_fn,
                vid_is_trimmed=True, # NOTE: assume we are always working with trimmed videos for agent
            )
        log_event(stage="agent", event="_load_and_verify_data",
                  msg=f"mapped keyframe indices from question annotation to subsampled video: {self.keyframe_idxs_in_subspl_video} -> {self.keyframe_idxs_for_question}",
                  meta={
                      "keyframe_idxs_for_question": self.keyframe_idxs_for_question,
                      "keyframe_idxs_in_subspl_video": self.keyframe_idxs_in_subspl_video,
                  })

        self.frame_fns = [
            osp.join(self.frame_dir, mid_path, f"{idx}.jpg") for idx in self.keyframe_idxs_for_question]
        self.prompt_masks = dict()
        with open(osp.join(self.prompt_masks_dir, mid_path, f"{self.video_id}.json")) as f:
            # import ipdb; ipdb.set_trace()
            all_prompt_masks = json.load(f)["manual"]
            for idx, keyframe_idx in enumerate(self.keyframe_idxs_for_question):
                if str(keyframe_idx) in all_prompt_masks:
                    self.prompt_masks[self.keyframe_idxs_in_subspl_video[idx]] = all_prompt_masks[str(keyframe_idx)]
        # logger.info(
        #     f"Loaded {len(self.prompt_masks)} prompt masks for video {self.video_id}",
        # )
        log_event(stage="agent", event="_load_and_verify_data",
                    msg=f"loaded {len(self.prompt_masks)} prompt masks for video {self.video_id}",
                    meta={
                        "num_prompt_masks": len(self.prompt_masks),
                    })

        # Handle jumble map if present for tracking questions
        self.question_jumble_map = None
        if "jumble_map" in self.question_json:
            self.question_jumble_map = dict()
            if isinstance(self.question_json["jumble_map"], list):
                for jm in self.question_json["jumble_map"]:
                    self.question_jumble_map.update(jm)
            elif isinstance(self.question_json["jumble_map"], dict):
                self.question_jumble_map = self.question_json["jumble_map"]
            # to be safe, convert all keys/vals to str
            self.question_jumble_map = {str(k): str(v) for k, v in self.question_jumble_map.items()}
            self._jumble_masks_for_question(jumble_map=self.question_jumble_map)
        
        
        # which frame index in the sub-sampled index is to be jumbled?
        self.jumbled_keyframe_idxs_in_subspl_video = None
        if len(self.keyframe_idxs_in_subspl_video) == 2:
            # NOTE: handling tracking cases
            self.jumbled_keyframe_idxs_in_subspl_video = copy.deepcopy(self.keyframe_idxs_in_subspl_video[1:])

        self.cached_video_segs_path = osp.join(self.cached_masks_dir, mid_path)

        # import ipdb; ipdb.set_trace()
        assert osp.exists(self.video_fn), f"Video file does not exist: {self.video_fn}"
        assert osp.exists(self.subspl_video_fn), f"Video file does not exist: {self.subspl_video_fn}"
        assert all([osp.exists(fn) for fn in self.frame_fns]), \
            f"Some prompt frame files for question do not exist: {self.frame_fns}"
        assert len(self.prompt_masks) == len(self.keyframe_idxs_for_question), \
            f"Some masks are missing: {self.prompt_masks.keys()} vs {self.keyframe_idxs_for_question}"
        assert len(self.metadata) == len(self.keyframe_idxs_for_question), \
            f"Some metadata are missing: {self.metadata.keys()} vs {self.keyframe_idxs_for_question}"
        assert osp.exists(self.cached_video_segs_path), \
            f"Cached video segments path does not exist: {self.cached_video_segs_path}"

        self.qinfo = convert_yaml_to_json(question_fn, add_abstain=True)
        # logger.info("Question context prepared for orchestrator consumption")
        log_event(stage="agent", event="_load_and_verify_data", msg="question context prepared for orchestrator consumption")
    
    def _jumble_masks_for_question(
        self,
        jumble_map: Dict[str, str],
    ):
        """
        Use the jumble map provided by the question to jumble the prompt masks.
        Args:
            jumble_map (Dict[str, str]): The jumble map.
        
        """
        tmp = {
            part_id: self.prompt_masks[self.keyframe_idxs_in_subspl_video[-1]][jumble_map[part_id]]
            for part_id in jumble_map if jumble_map[part_id] in self.prompt_masks[self.keyframe_idxs_in_subspl_video[-1]]
        }
        self.prompt_masks[self.keyframe_idxs_in_subspl_video[-1]] = tmp


    def _create_prompt_for_orchestrator(
        self,
        question_str: str,
        conv_template: str = "agent_conversation",
        api_prompt_version: str = "v1",
        print_prompt: bool = False,
    ) -> List[Dict[str, Any]]:
        """Create a prompt for the orchestrator.

        Args:
            question_str (str): The question string.
            conv_template (str, optional): The conversation template file to use. Defaults to "agent_conversation".
                If complete filename is not given, it will search in `src/tva/media/prompts/`.
            print_prompt (bool, optional): Whether to print the prompt. Defaults to False.
            api_prompt_version (str, optional): Version of the API prompt to use. Defaults to "v1".
        Returns:
            List[Dict[str, Any]]: The prompt as a list of dictionaries.
        """
        # logger.info(f"Building orchestrator prompt using template {conv_template}")
        log_event(stage="agent", event="create_prompt", msg=f"building orchestrator prompt using template {conv_template}")
        conv_template_fn = None
        # import ipdb; ipdb.set_trace()
        if os.path.exists(conv_template):
            conv_template_fn = conv_template
        elif os.path.exists(osp.join(root, "src", "tva", "media", "prompts", f"{conv_template}.yaml" if not conv_template.endswith(".yaml") else conv_template)):
            conv_template_fn = osp.join(root, "src", "tva", "media", "prompts", f"{conv_template}.yaml" if not conv_template.endswith(".yaml") else conv_template)
        else:
            raise FileNotFoundError(f"Conversation template file not found: {conv_template}")
        
        with open(conv_template_fn, 'r') as f:
            conv_template = yaml.safe_load(f)
        
        prompt = list()
        msg_container = {
            "tag": "message", 
            # dummy tag by default, merging all messages apart from
            # potential system prompt into one message to avoid unexpected
            # whitespace handling by the LLM processor
            "type": "text",
            "content": "",      
        }
        for msg_idx, msg in enumerate(conv_template):
            if msg["tag"] == "task_instruction":
                # NOTE: assumption is that there is only a single task_instruction message
                # either at the end or start of the conversation template
                if self.change_sys_prompt:
                    prompt.append(msg)
                else:
                    if msg_idx == 0:
                        prompt.append(msg_container)
                    prompt[-1]["content"] += msg["content"]
                continue
            
            elif msg["tag"] == "question":
                msg["content"] = question_str
                
            elif msg["tag"] == "api_description":
                if msg["type"] == "text_file":
                    content_fn = None
                    # import ipdb; ipdb.set_trace()
                    if osp.exists(msg["content"]):
                        content_fn = msg["content"]
                        
                    elif osp.exists(osp.join(root, "src", "tva", "media", "prompts", 
                                             f"{msg['content']}{'.' + api_prompt_version + '.txt' if api_prompt_version else '.txt'}" 
                                             if not msg['content'].endswith(".txt") 
                                             else msg['content']
                    )):
                        content_fn = osp.join(root, "src", "tva", "media", "prompts", 
                                              f"{msg['content']}{'.' + api_prompt_version + '.txt' if api_prompt_version else '.txt'}" 
                                              if not msg['content'].endswith(".txt") 
                                              else msg['content'])
                    else:
                        raise FileNotFoundError(f"API description file not found: {msg['content']}")
                    with open(content_fn, 'r') as f:
                        msg["content"] = f.read()
                elif msg["type"] == "text":
                    msg["content"] = msg["content"]
                else:
                    raise ValueError(f"Unsupported api_description type: {msg['type']}")
                
            if msg_idx == 0 or prompt[-1]["tag"] == "task_instruction":
                prompt.append(msg_container)
            prompt[-1]["content"] += msg["content"]
                
        if print_prompt:
            # print(yaml.dump(prompt))
            with open(osp.join(self.cache_dir, "orchestrator_prompt_debug.yaml"), 'w') as f:
                yaml.dump(prompt, f)
            log_event(stage="agent", event="create_prompt", msg="orchestrator prompt printed to cache directory for debugging",
                      file_path=osp.join(self.cache_dir, "orchestrator_prompt_debug.yaml"))

        # logger.info(f"Prompt constructed with {len(prompt)} message blocks")
        log_event(stage="agent", event="create_prompt", msg=f"prompt constructed with {len(prompt)} message blocks",
                  meta={
                      "num_message_blocks": len(prompt),
                  })
        return prompt

    def get_code(
        self,
        question_fn: str = None,
        question_json: Optional[Dict] = None,
        conv_template: str = "agent_conversation",
        conv_template_extra: Dict[str, Any] = {
            "tracking": "agent_conversation_tracking"
        },
        api_prompt_version: str = "v1",
        print_prompt: bool = False,
        cached_qinfo: Optional[Dict] = None,
    ):
        """
        Generate code for the given question.
        
        Args:
            question_fn (str, optional): Path to the question YAML file. Defaults to None.
                Only used if `question_json` is None.
            question_json (Dict, optional): Question in JSON format. Defaults to None.
            conv_template (str, optional): Conversation template to use. Defaults to "agent_conversation".
            conv_template_extra (Dict[str, Any], optional): Extra conversation templates for specific categories. Defaults to {
                "tracking": "agent_conversation_tracking"
            }
            api_prompt_version (str, optional): Version of the API prompt to use. Defaults to "v1".
            print_prompt (bool, optional): Whether to print the prompt. Defaults to False.
            cached_qinfo (Dict, optional): Cached question information to use. Defaults to None.
        Returns:
            str: The generated code.
        """
        
        if question_json is None and question_fn is None:
            raise ValueError("Either question_fn or question_json must be provided.")
        
        log_event(stage="agent", event="load_data", msg="loading and verifying question data")
        self._load_and_verify_data(question_fn, question_json)
        if cached_qinfo is not None:
            self.qinfo = cached_qinfo
        with open(osp.join(self.cache_dir, "qinfo.json"), 'w') as f:
            json.dump(self.qinfo, f, indent=4)
        log_event(stage="agent", event="load_data", msg="question data loaded and verified", file_path=osp.join(self.cache_dir, "qinfo.json"))
        
        log_event(stage="agent", event="create_prompt", msg="Creating prompt for orchestrator")
        # import ipdb; ipdb.set_trace()
        if self.question_json["question_category"] in conv_template_extra:
            conv_template = conv_template_extra[self.question_json["question_category"]]
            log_event(stage="agent", event="create_prompt", 
                      msg=f"Using conversation template {conv_template} for question category {self.question_json['question_category']}")

        prompt = self._create_prompt_for_orchestrator(
            question_str=self.qinfo["question"]["qstr"],
            conv_template=conv_template,
            print_prompt=print_prompt,
            api_prompt_version=api_prompt_version,
        )
        
        with open(osp.join(self.cache_dir, "orchestrator_prompt.yaml"), 'w') as f:
            yaml.dump(prompt, f)
        log_event(stage="agent", event="get_code", msg="prompt created and saved", 
                  file_path=osp.join(self.cache_dir, "orchestrator_prompt.yaml"))
        # import ipdb; ipdb.set_trace()
        log_event(stage="agent", event="get_code", msg="calling orchestrator for code generation")
        response = self._call_orchestrator(prompt)
        with open(osp.join(self.cache_dir, "orchestrator_response.json"), 'w') as f:
            json.dump(response, f, indent=4)
        log_event(stage="agent", event="get_code", msg="orchestrator response received and saved", 
                  file_path=osp.join(self.cache_dir, "orchestrator_response.json"))
        with open(osp.join(self.cache_dir, "generated_code.py"), 'w') as f:
            f.write(json.loads(response["response"])["code"])
        log_event(stage="agent", event="get_code", msg="generated code saved", 
                  file_path=osp.join(self.cache_dir, "generated_code.py"))
        
        if "thoughts" in response:
            with open(osp.join(self.cache_dir, "orchestrator_thoughts.txt"), 'w') as f:
                f.write(response["thoughts"] if isinstance(response["thoughts"], str) else "\n".join(response["thoughts"]))
            log_event(stage="agent", event="get_code", msg="orchestrator thoughts saved", 
                      file_path=osp.join(self.cache_dir, "orchestrator_thoughts.txt"))
        
        log_event(stage="agent", event="get_code", msg="orchestrator call complete")

        return response
    
    def _call_orchestrator(
        self,
        prompt: List[Dict[str, Any]],
    ) -> str:
        response = self.orchestrator(
            conversation=prompt
        )
        return response
    
    def execute_code(
        self,
        code: str,
        *,
        addl_params: Dict[str, Dict[str, Any]] | None = None,
        entrypoint: str = "execute_code",
        entrypoint_kwargs: Dict[str, Any] | None = None,
        extra_globals: Dict[str, Any] | None = None,
        safe_mode: bool = True,
    ) -> Dict[str, Any]:
        """Execute the given code string in a controlled environment.

        Args:
            code (str): The code string to execute.
            addl_params (Optional[dict], optional): A dictionary of method names to parameter
                dictionaries that will override the method call-time args/kwargs. Must contain
                a "method_params" key for each top-level key. Defaults to None.
                structure:
                <class_name>:
                    <arg1>: <value1>
                    <arg2>: <value2>
                    ...
                    "method_params":
                        <method_name>:
                            <param1>: <value1>
                            <param2>: <value2>
                            ...
            entrypoint (str, optional): The name of the function to call after executing
                the code. If None, no function is called. Defaults to "execute_code".
            entrypoint_kwargs (Optional[dict], optional): Keyword arguments to pass to
                the entrypoint function. Defaults to None.
            extra_globals (Optional[dict], optional): Additional global variables to
                include in the execution environment. Defaults to None.
            safe_mode (bool, optional): If True, restricts builtins to a safe subset.
                Defaults to True.
        Returns:
            Dict[str, Any]: A dictionary with keys:
                - "ok" (bool): Whether execution was successful.
                - "result" (Any): The result of the entrypoint function, if called.
                - "error" (Optional[str]): Error message if execution failed.
        """
        log_event(stage="agent", event="execute_code", msg=f"preparing to execute generated code with entrypoint: {entrypoint}")
        
        with open(osp.join(self.cache_dir, "tva_method_additional_params.json"), 'w') as f:
            json.dump(addl_params, f, indent=4)
        log_event(stage="__main__", event="main", msg="additional parameters for code execution saved", 
                file_path=osp.join(self.cache_dir, "tva_method_additional_params.json"))

        replace_constructors = dict()

        addl_params_to_use = addl_params or dict()

        for class_name in addl_params_to_use:
            if class_name not in ["ImagePatch", "VideoSegment"]:
                # logger.warning(f"Unknown class name in addl_params: {class_name}")
                log_event(stage="agent", event="execute_code", msg=f"unknown class name in addl_params: {class_name}")
                raise ValueError(f"Unknown class name in addl_params: {class_name}")
        
            replace_constructors[class_name] = {
                "partial": functools.partial(eval(class_name), **addl_params_to_use[class_name]),
                "name": f"{class_name.lower()}_ctor",
            }
        
        code = rewrite_constructors_to_partials(
            code_str=code,
            replacements={k: v["name"] for k, v in replace_constructors.items()},
        )
        log_event(stage="agent", event="execute_code", 
                  msg=f"rewrote constructors to partials for: {list(replace_constructors.keys())}")
        
        code_kwargs = dict()
        if self.question_category == "tracking":
            code_kwargs["video_fn"] = self.subspl_video_fn            
            code_kwargs["input_image_a"] = self.frame_fns[0]
            code_kwargs["input_image_b"] = self.frame_fns[1]
            
            code_kwargs["frame_idx_a"] = self.keyframe_idxs_in_subspl_video[0]
            code_kwargs["frame_idx_b"] = self.keyframe_idxs_in_subspl_video[1]

            # it is better for the orchestrator if the masks are decoded np.ndarrays
            # as it will rely less on external libs to handle RLE decoding and therefore
            # hopefully hallucinate less
            code_kwargs["masks_a"] = {k: v if isinstance(v, np.ndarray) else decode(v) for k, v in self.prompt_masks[self.keyframe_idxs_in_subspl_video[0]].items()}
            code_kwargs["masks_b"] = {k: v if isinstance(v, np.ndarray) else decode(v) for k, v in self.prompt_masks[self.keyframe_idxs_in_subspl_video[1]].items()}    

        else:
            code_kwargs["video_fn"] = self.subspl_video_fn
            code_kwargs["input_image"] = self.frame_fns[-1]
            code_kwargs["frame_idx"] = self.keyframe_idxs_in_subspl_video[-1]
            code_kwargs["masks"] = {k: v if isinstance(v, np.ndarray) else decode(v) for k, v in self.prompt_masks[self.keyframe_idxs_in_subspl_video[-1]].items()}

        code_kwargs.update(entrypoint_kwargs or dict())
        log_event(stage="agent", event="execute_code", 
                  msg=f"prepared code execution kwargs: {list(code_kwargs.keys())}")

        # 2) Prepare a globals dict for execution
        #    Include common libs you expect the generated code to use.
        #    Also include original classes in case code does isinstance/ImagePatch checks.
        safe_builtins = {
            "abs": builtins.abs,
            "all": builtins.all,
            "any": builtins.any,
            "bool": builtins.bool,
            "dict": builtins.dict,
            "enumerate": builtins.enumerate,
            "float": builtins.float,
            "int": builtins.int,
            "len": builtins.len,
            "list": builtins.list,
            "map": builtins.map,
            "max": builtins.max,
            "min": builtins.min,
            "print": builtins.print,
            "range": builtins.range,
            "str": builtins.str,
            "sum": builtins.sum,
            "zip": builtins.zip,
            "set": builtins.set,
        }

        exec_globals: Dict[str, Any] = {
            "__name__": "__generated__",   # so `if __name__ == "__main__"` won’t run
            "__package__": None,
            # Partials exposed under the names your AST rewriter uses:
            **{v["name"]: v["partial"] for k, v in replace_constructors.items()},
            # It can still be useful to expose the real classes:
            "ImagePatch": ImagePatch,
            "VideoSegment": VideoSegment,
            # Common libs (adjust to your needs):
            "np": np,
            "numpy": np,
            "os": os,
            "osp": osp,
            "json": json,
            "logger": logger,
            "PIL": PIL,
            "Image": Image,
            "typing": typing,
            "Dict": typing.Dict,
            "List": typing.List,
            "Optional": typing.Optional,
            "Any": typing.Any,
            "Union": typing.Union,
            "functools": functools,
            "traceback": traceback,
            "pd": pd,
            "pandas": pd,
        }
        # Optional: add more project utilities that LLM code may reference
        exec_globals.update(extra_globals or dict())

        # Builtins policy
        if safe_mode:
            exec_globals["__builtins__"] = safe_builtins
        else:
            exec_globals["__builtins__"] = builtins.__dict__
        # import ipdb; ipdb.set_trace()
        # 3) Exec the code
        try:
            compiled = compile(code, filename="<generated>", mode="exec")
            exec(compiled, exec_globals, exec_globals)
        except Exception as e:
            return {
                "ok": False,
                "result": None,
                "error": "EXEC_ERROR:\n" + traceback.format_exc(),
            }

        log_event(stage="agent", event="execute_code", 
                  msg=f"code executed successfully, looking for entrypoint: {entrypoint}")

        # 4) Optionally call a known entrypoint
        if entrypoint:
            fn = exec_globals.get(entrypoint)
            if callable(fn):
                try:
                    result = fn(**code_kwargs)
                    return {"ok": True, "result": result, "error": None}
                except Exception:
                    return {
                        "ok": False,
                        "result": None,
                        "error": "ENTRYPOINT_ERROR:\n" + traceback.format_exc(),
                    }
            else:
                # No entrypoint present; treat exec success as OK
                return {"ok": True, "result": None, "error": None}
        else:
            # Just exec, no call
            return {"ok": True, "result": None, "error": None}



    
def main():
    
    from src.eval.models.gemini import Gemini
    
    args = parse_args()

    if args.questions_dir:
        if args.question_fn:
            raise ValueError("Use either --question-fn or --questions-dir, not both.")

        base_cli_args: List[str] = [
            "--data-root",
            str(args.data_root),
            "--cache-dir-root",
            str(args.cache_dir_root),
            "--api-prompt-version",
            str(args.api_prompt_version),
        ]
        if args.load_code:
            base_cli_args.extend(["--load-code", str(args.load_code)])
        if args.questions_jsonl_fn:
            base_cli_args.extend(["--questions-jsonl-fn", str(args.questions_jsonl_fn)])

        log_event(
            stage="__main__",
            event="run_all_questions",
            msg="Launching TVAAgent over all questions in directory",
            meta={
                "questions_dir": args.questions_dir,
                "max_workers": args.max_workers,
                "fail_fast": args.fail_fast,
            },
        )

        results = run_all_questions(
            question_dir=args.questions_dir,
            base_cli_args=base_cli_args,
            max_workers=args.max_workers,
            fail_fast=args.fail_fast,
        )

        failures = {path: code for path, code in results.items() if code != 0}

        log_event(
            stage="__main__",
            event="run_all_questions",
            msg="Completed TVAAgent batch runs",
            meta={
                "questions_dir": args.questions_dir,
                "total_runs": len(results),
                "failed_runs": len(failures),
            },
        )

        if failures:
            log_event(
                stage="__main__",
                event="run_all_questions",
                msg="One or more TVAAgent runs failed",
                meta={"failures": failures},
            )
            raise SystemExit(1)

        return
    
    data_dir = args.data_root
    video_dir = osp.join(data_dir, "videos")
    subspl_video_dir = osp.join(data_dir, "videos", "subsampled", "subspl-by-4")
    cached_masks_dir = osp.join(root, "tmp", "tva_vid_segs", "subspl-by-4")
    subspl_factor_for_videos = 4
    frame_dir = osp.join(data_dir, "rgb-frames")
    metadata_dir = osp.join(data_dir, "frames-metadata")
    
    manual_data_dir = osp.join(root, "data")
    prompt_masks_dir = osp.join(manual_data_dir, "segmentation-masks")

    question_fn = args.question_fn or osp.join(manual_data_dir, "questions", "yamls", "032.yaml")
    questions_jsonl_fn = args.questions_jsonl_fn
    cached_qinfo = None
    if questions_jsonl_fn is not None and osp.exists(questions_jsonl_fn):
        with open(questions_jsonl_fn, 'r') as f:
            for line_idx, line in enumerate(f):
                line_json = json.loads(line.strip())
                if line_idx == osp.basename(args.question_fn).split(".")[0]:
                    cached_qinfo = line_json
                    break
    if not osp.exists(question_fn):
        question_fn = osp.join(manual_data_dir, "questions", "yamls", args.question_fn)
    assert osp.exists(question_fn), f"Question file does not exist: {question_fn}"
    cache_dir = f"{args.cache_dir_root}/{osp.basename(question_fn).split('.')[0]}"

    if osp.exists(cache_dir):
        if osp.exists(osp.join(cache_dir, "execution_result.json")):
            with open(osp.join(cache_dir, "execution_result.json"), 'r') as f:
                execution_result = json.load(f)
            if execution_result["ok"]:
                return
            else:
                shutil.rmtree(cache_dir)
        else:
            shutil.rmtree(cache_dir)


    os.makedirs(cache_dir, exist_ok=True)
    logs_path = osp.join(cache_dir, "logs.jsonl")
    logging_setup(logs_path)

    # import ipdb; ipdb.set_trace()
    model = Gemini(
        model_name="gemini-2.5-flash",
        response_schema="GeminiCode",
        clear_cache_at_init=False,
        tmpdir=osp.join(root, "tmp", "gemini_cache"),
    )
    # logger.info(f"Initialized orchestrator model: {model.__class__.__name__}")
    log_event(stage="__main__", event="main", msg=f"initialized orchestrator model: {model.__class__.__name__}",
              meta={
                  "model_version": model.model_version,
                  "response_schema": "GeminiCode",
              })
    
    morpheus = TVAAgent(
        orchestrator=model,
        video_dir=video_dir,
        frame_dir=frame_dir,
        subspl_video_dir=subspl_video_dir,
        cached_masks_dir=cached_masks_dir,
        prompt_masks_dir=prompt_masks_dir,
        metadata_dir=metadata_dir,
        cache_dir=cache_dir,
        change_sys_prompt_for_orchestrator=True,
        subspl_factor_for_videos=subspl_factor_for_videos,
    )
    # logger.info("Constructed TVAAgent instance")
    log_event(stage="__main__", event="main", msg="constructed TVAAgent instance")

    
    
    if args.load_code is not None:
        with open(args.load_code, "r") as f:
            response = json.loads(f.read())
        code_str = json.loads(response["response"])["code"]
        # logger.info(f"Loaded code from {args.load_code}")
        log_event(stage="__main__", event="main", msg=f"loaded code from {args.load_code}")
        with open(question_fn, 'r') as f:
            question_json = yaml.safe_load(f)
        morpheus.question_json = question_json
        morpheus._load_and_verify_data(question_fn=question_fn, question_json=question_json)
    else:
        log_event(stage="__main__", event="main", msg="generating code using orchestrator")
        response = morpheus.get_code(
            question_fn=question_fn,
            conv_template="agent_conversation",
            print_prompt=False,
            api_prompt_version=args.api_prompt_version,
            cached_qinfo=cached_qinfo,
        )

    code_str = json.loads(response["response"])["code"]
    
    
    # import ipdb; ipdb.set_trace()
    # logger.info(f"""===== GENERATED CODE =====
    # {code_str}
    # """)
    # if "thoughts" in response:
    #     logger.info(f"""===== REASONING =====
    #     {response['thoughts'][0] if isinstance(response['thoughts'], list) else response['thoughts']}
    #     """)

    addl_params = {
        f"{ImagePatch.__name__}": {
            "cache_dir": cache_dir,
            "method_params": {
                "get_part_visibility": {"visibility_threshold": 0.0},
                "vlm_query": {
                    "vlm_model": "qwen25_vl",
                    "prompt_template": "default",
                    "model_init_args": {
                        "model_name": "Qwen/Qwen2.5-VL-32B-Instruct",
                        "generate_config": {"temperature": 0.0},
                        "forward_pipeline": "vllm_online"
                    },
                },
                "check_part_connectivity": {
                    # "vlm_model": "gemini",
                    "vlm_model": "qwen25_vl",
                    "model_init_args": {
                        # "model_name": "gemini-2.5-flash",
                        # "generate_config": {"temperature": 0.0}
                        "model_name": "Qwen/Qwen2.5-VL-32B-Instruct",
                        "generate_config": {"temperature": 0.0},
                        "forward_pipeline": "vllm_online",
                        "add_vision_id": True,
                    },
                },
                "get_all_connected_part_pairs": {
                    "vlm_model": "qwen25_vl",
                    "model_init_args": {
                        "model_name": "Qwen/Qwen2.5-VL-32B-Instruct",
                        "generate_config": {"temperature": 0.0},
                        "forward_pipeline": "vllm_online",
                        "add_vision_id": True,
                    },
                    "print_msgs": True,
                },
                "track_and_find_all_connected_pairs": {
                    "vlm_model": "qwen25_vl",
                    "model_init_args": {
                        "model_name": "Qwen/Qwen2.5-VL-32B-Instruct",
                        "generate_config": {"temperature": 0.0},
                        "forward_pipeline": "vllm_online",
                        "add_vision_id": True,
                    },
                }
            }
        },

        f"{VideoSegment.__name__}": {
            "cache_dir": cache_dir,
            "multi_mask": False,
            "cached_seg_path": morpheus.cached_video_segs_path,
            "subspl_factor": morpheus.subspl_factor_for_videos,
            "method_params": {
                "track_object_segments_in_video": {
                    "non_overlap_masks": True,
                    "vos_model": "sam2",
                    "prompt_mode": "mask",
                    "device": "cuda",
                    "debug_stride": None,
                    "incremental_dump": True,
                    "overwrite_existing": False,
                    "overwrite_cache": False,
                    "is_frame_idx_subspled": True,
                },
            }
        }
    }

    # import ipdb; ipdb.set_trace()

    morpheus_execute_result = morpheus.execute_code(
        code=code_str,
        addl_params=addl_params,
        entrypoint="execute_code",
        entrypoint_kwargs=None,
        extra_globals=None,
        safe_mode=False,
    )
    with open(osp.join(cache_dir, "execution_result.json"), 'w') as f:
        json.dump(morpheus_execute_result, f, indent=4)
    log_event(stage="agent", event="execute_code", msg="code execution complete", 
              file_path=osp.join(cache_dir, "execution_result.json"))
    # logger.info(f"""===== EXECUTION RESULT =====""")
    # logger.info(f"""
    #     status: {morpheus_execute_result['ok']}
    #     result:\n {morpheus_execute_result['result']}
    #     error:\n {morpheus_execute_result['error']}
    # """)
    # print("===== EXECUTION RESULT =====")
    # print(morpheus_execute_result)


def run_all_questions(
    question_dir: str,
    *,
    base_cli_args: Optional[Sequence[str]] = None,
    max_workers: Optional[int] = None,
    fail_fast: bool = False,
) -> Dict[str, int]:
    """Launch ``main`` in parallel for every ``.yaml`` file in ``question_dir``.

    Each run executes in its own subprocess so that generated code, caches,
    and logging mirror the normal CLI behavior of ``main``. The return value is a
    mapping from absolute YAML paths to the respective process return codes.

    Args:
        question_dir: Directory containing question YAML files.
        base_cli_args: Extra command-line arguments (e.g., ``["--api-prompt-version", "v2"]``)
            applied to every spawned process. ``--question-fn`` must not be supplied.
        max_workers: Upper bound on concurrent subprocesses. Defaults to the
            smaller of CPU count and number of YAML files.
        fail_fast: If True, cancel outstanding jobs and raise as soon as a run
            exits with a non-zero status.
    """

    question_path = Path(question_dir)
    if not question_path.is_dir():
        raise ValueError(f"Expected a directory of questions, got: {question_dir}")

    yaml_files = sorted(question_path.glob("*.yaml"))
    if not yaml_files:
        raise ValueError(f"No .yaml files found under {question_dir}")

    extra_args: List[str] = list(base_cli_args) if base_cli_args else []
    if "--question-fn" in extra_args:
        raise ValueError("base_cli_args should not include --question-fn")

    worker_count = max_workers or min(len(yaml_files), max(1, os.cpu_count() or 1))
    results: Dict[str, int] = {}

    def _launch(question_file: Path) -> Tuple[str, int]:
        cmd = [sys.executable, __file__]
        cmd.extend(extra_args)
        cmd.extend(["--question-fn", str(question_file)])
        completed = subprocess.run(cmd, check=False)
        return str(question_file), completed.returncode

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {executor.submit(_launch, path): path for path in yaml_files}
        try:
            for future in as_completed(future_map):
                q_path, returncode = future.result()
                results[q_path] = returncode
                if returncode != 0 and fail_fast:
                    for pending in future_map:
                        if not pending.done():
                            pending.cancel()
                    raise RuntimeError(
                        f"Run failed for {q_path} with exit code {returncode}"
                    )
        finally:
            # Ensure remaining futures see cancellation if we exit early.
            if fail_fast:
                for pending in future_map:
                    if not pending.done():
                        pending.cancel()

    return results


def parse_args():

    import argparse
    
    parser = argparse.ArgumentParser(description="TVAAgent main entry point")
    parser.add_argument(
        "--data-root",
        type=str,
        default=osp.join(root, "data"),
        help="Root directory for the IKEA Manuals at Work dataset."
    )
    parser.add_argument(
        "--question-fn",
        type=str,
        default=None,
        help="Path to the question YAML file."
    )
    parser.add_argument(
        "--cache-dir-root",
        type=str,
        default=osp.join(root, "tmp", "tva_agent_cache"),
        help="Directory to use for caching intermediate results."
    )

    parser.add_argument(
        "--api-prompt-version",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help="Version of the API prompt to use."
    )

    parser.add_argument(
        "--load-code",
        type=str,
        default=None,
        help="Path to a JSON file containing previously generated code to load instead of generating new code."
    )

    parser.add_argument(
        "--questions-dir",
        type=str,
        default=None,
        help="Directory containing question YAML files to process in parallel."
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel TVAAgent runs when using --questions-dir."
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop launching new runs and exit on first failure when using --questions-dir."
    )

    parser.add_argument(
        "--questions-jsonl-fn",
        type=str,
        default=osp.join(root, "data", "questions", "questions.jsonl"),
        help="Path to a JSONL file containing multiple questions."
    )
    
    # 032.yaml, 064.yaml
    return parser.parse_args()
    


    

    
    
    
if __name__ == "__main__":
    main()
