import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from typing import Dict, Any, Literal, Tuple, Optional
import os
import re
import os.path as osp
import json
import math
import time
import tempfile
import mimetypes

import mediapy
import numpy as np

from google import genai
from google.genai import types
from google.genai.types import HttpOptions
from google.cloud import storage
from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel as PydanticBaseModel

from loguru import logger as eval_logger

from src.eval.models.base_model import BaseModel
from src.benchmark.lmmeval.models.utils import retry_with_exponential_backoff
from src.eval.prompts.templates.questions.common import calculate_sha256


MAX_TRIES = 5

def upload_video_if_absent(bucket_name: str, local_path: str, dest_path: str | None = None) -> str:
    """
    Uploads a video to GCS only if the object does NOT already exist.
    Returns the gs:// path
    
    Args:
        bucket_name (str): The name of the GCS bucket.
        local_path (str): The local path to the video file.
        dest_path (str | None, optional): The destination path in the bucket. Defaults to None.
    Returns:
        str: The gs:// path of the uploaded video or None if skipped.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    object_name = dest_path
    if object_name is None:
        object_name = osp.join(
            "media_cache",
            local_path.split("/")[-2],
            osp.basename(local_path)
        )
    blob = bucket.blob(object_name)

    # Cheap existence check (HEAD request on the object)
    if blob.exists():
        eval_logger.info(f"Skipped (already exists): gs://{bucket_name}/{object_name}")
        return f"gs://{bucket_name}/{object_name}"

    # Best-effort content type
    ctype, _ = mimetypes.guess_type(local_path)
    if not ctype:
        dom = local_path.split(".")[-1]
        if dom in ["png", "jpg", "jpeg"]:
            ctype = f"image/{dom}"
        elif dom in ["mp4", "mov"]:
            ctype = f"video/{dom}"
            
    if ctype == "image/jpg":
        # normalizing for consistency with API
        ctype = "image/jpeg"

    # Resumable upload for large files (must be multiple of 256 KB)
    blob.chunk_size = 8 * 1024 * 1024

    blob.upload_from_filename(local_path, content_type=ctype)
    eval_logger.info(f"Uploaded: gs://{bucket_name}/{object_name}")
    return f"gs://{bucket_name}/{object_name}"

RESPONSE_SCHEMA_REGISTRY: dict[str, type] = dict()
def register_response_schema(schema: Any):
    RESPONSE_SCHEMA_REGISTRY[schema.__name__] = schema

def resolve_response_schema(schema_name: Optional[str]) -> Any:
    if schema_name and schema_name in RESPONSE_SCHEMA_REGISTRY:
        return RESPONSE_SCHEMA_REGISTRY[schema_name]
    elif schema_name is None:
        return None
    else:
        raise ValueError(f"Response schema {schema_name} not found in registry.")
    

def process_video_if_needed(
        video_path: str, 
        max_duration: int, 
    tmpdir: str = "tmp/gemini",
        target_fps: int = None,
        num_init_frames_to_preserve: int = None,
    ) -> str:
    """Process the video to ensure it meets the fps requirements.

    Args:
        video_path (str): Path to the input video.
        max_duration (int): Maximum duration allowed in seconds.
        tmpdir (str): Directory to store temporary processed video.

    Returns:
        str: Path to the processed video.
    """
    if osp.exists(osp.join(tmpdir, osp.basename(video_path).split(".")[0] + f"_processed_{max_duration}fps.mp4")):
        return osp.join(tmpdir, osp.basename(video_path).split(".")[0] + f"_processed_{max_duration}fps.mp4")
    # get number of frames in the video
    vid_frames = mediapy.read_video(video_path)
    num_frames = len(vid_frames)
    num_init_frames_to_preserve = 0 if num_init_frames_to_preserve is None else num_init_frames_to_preserve
    # import ipdb; ipdb.set_trace()   
    if num_frames > max_duration:
        # NOTE: assumption: input video fps is 1
        target_fps = max(
            math.ceil((num_frames - num_init_frames_to_preserve) / (max_duration - num_init_frames_to_preserve)), 
            target_fps if target_fps is not None else 1
        )
        # Process the video to reduce fps
        if num_init_frames_to_preserve > 0:
            # ensures that the frames to be preserved are displayed for 1 second each
            # ensuring their sampling as per the API (Gemini processes videos at 1 FPS)
            vid_frames = np.vstack(
                np.repeat(vid_frames[:num_init_frames_to_preserve], target_fps, axis=0),
                vid_frames[num_init_frames_to_preserve:]
            )
            assert num_init_frames_to_preserve + (num_frames - num_init_frames_to_preserve) // target_fps <= max_duration, \
                f"After preserving the initial {num_init_frames_to_preserve} frames, the \
                    video duration ({num_init_frames_to_preserve + (num_frames - num_init_frames_to_preserve) // target_fps}) exceeds \
                    the maximum duration {max_duration}. \
                    Check the target_fps ({target_fps}) or num_init_frames_to_preserve ({num_init_frames_to_preserve})."
        tmpfile = osp.join(tmpdir, osp.basename(video_path).split(".")[0] + f"_processed_{target_fps}fps.mp4")
        if not osp.exists(tmpfile):
            eval_logger.info(f"Processing video {video_path} to {target_fps} fps.")
            mediapy.write_video(tmpfile, vid_frames, fps=target_fps)
        return tmpfile
            
    return video_path

@register_response_schema
class GeminiAnswer(PydanticBaseModel):
    
    answer: str

@register_response_schema
class GeminiCode(PydanticBaseModel):
    
    code: str
    
class Gemini(BaseModel):
    
    def __init__(
        self, 
        model_name: Literal["gemini-2.5-flash", "gemini-2.5-pro"] = "gemini-2.5-pro",
        api_key: str = None,
        thinking_config: Dict[str, Any] = {"thinking_budget": -1, "include_thoughts": True},
        generate_config: Dict[str, Any] = {"temperature": 0.0, "media_resolution": "high", "seed": 42},
        use_vertexai: bool = False,
        gs_bucket: str = "vlm_4d_bench_vids",
        video_fps: int = None,
        api_version: str = "v1",
        response_schema: str | None = "GeminiAnswer",
        tmpdir: str = "tmp/gemini",
        clear_cache_at_init: bool = True,
        use_paid_api: bool = False,
        **kwargs
    ):
        """Gemini model wrapper for Google GenAI.

        For `thinking_budget` in `thinking_config`, a value of -1 means dynamic thinking,
        which allows the model to decide how long to think before generating an answer.
        For `temperature` in `generate_config`, a value of 0.0 means deterministic generation.
        
        TODO: is there a way I can persist the files uploaded to the client across multiple runs
              of the model?
        
        Args:
            model_name (Literal["gemini-2.5-flash", "gemini-2.5-pro"], optional): model version. Defaults to "gemini-2.5-pro".
            api_key (str, optional): API key for authentication. Defaults to None.
            thinking_config (Dict[str, Any], optional): Configuration for thinking. Defaults to {"thinking_budget": -1}.
            generate_config (Dict[str, Any], optional): Configuration for generation. Defaults to {"temperature": 0.0}.
            use_vertexai (bool, optional): Whether to use Vertex AI. Defaults to False.
            gs_bucket (str, optional): google cloud storage bucket for vertex endpoint
            video_fps (int, optional): Frames per second for video processing. Defaults to 1.
                Used when we have to re-process videos to fit within duration limits.
            api_version (str, optional): API version for Vertex AI. Defaults to "v1". 
                Can be ["v1", "v1alpha1", "v1beta1"]
            tmpdir (str, optional): Temporary directory for processing. Defaults to "tmp/gemini".
            use_paid_api (bool, optional): Whether to use the paid API key. Defaults to False.
            **kwargs: Additional keyword arguments.
        """
        
        super().__init__(**kwargs)
        self.model_version = model_name
        self.api_key = api_key
        self.thinking_config = thinking_config
        self.generate_config = generate_config
        if self.generate_config.get("media_resolution") is not None:
            if self.generate_config["media_resolution"] not in ["high", "medium", "low"]:
                raise ValueError("media_resolution must be one of 'high', 'medium', or 'low'")
            media_resolution_map = {
                "high": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
                "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
                "low": types.MediaResolution.MEDIA_RESOLUTION_LOW
            }
            self.generate_config["media_resolution"] = media_resolution_map[
                self.generate_config.get("media_resolution", "high")]
        # source https://ai.google.dev/gemini-api/docs/video-understanding#technical-details-video
        self.max_duration_of_videos = int(3600 * 3 * 0.8) \
            if "media_resolution" in self.generate_config and \
                self.generate_config["media_resolution"] == types.MediaResolution.MEDIA_RESOLUTION_LOW \
                    else int(3600 * 1 * 0.8) # seconds
        # import ipdb; ipdb.set_trace()
        self.tmpdir = tmpdir
        
        self.api_version = api_version
        self.video_fps = video_fps
        self.use_vertexai = use_vertexai
        # import ipdb; ipdb.set_trace()

        self.response_schema = resolve_response_schema(response_schema)

        self.client = None
        if not use_vertexai:
            self.client = genai.Client(
                vertexai=use_vertexai,
                api_key=os.getenv("GEMINI_API_KEY" if not use_paid_api else "GEMINI_PAID_API_KEY") if api_key is None else api_key,
                # http_options=HttpOptions(api_version="v1"),
            )
        else:
            self.client = genai.Client(
                vertexai=use_vertexai,
                http_options=HttpOptions(api_version=api_version),
                project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                location=os.getenv("GOOGLE_CLOUD_LOCATION"),
            )
        self._files_cache = dict()
        self.gs_bucket = gs_bucket
        # NOTE: when using this here, we do not need to edit the _files_cache because nothing is uploaded
        # yet that will be cleared by this
        if clear_cache_at_init:
            self.clear_files_cache()
        
    @retry_with_exponential_backoff
    def generate_content_with_backoff(self, **kwargs):
        # import ipdb; ipdb.set_trace()
        return self.client.models.generate_content(**kwargs)
    
    def check_files_cache(
        self,
        flush_size_in_bytes: int = (2e+9) * 0.7,
    ):
        
        if self.use_vertexai:
            return
        cur_size_in_bytes = 0
        size_dict = dict()
        for fn in self._files_cache:
            cache_file_name = self._files_cache[fn]
            f = self.client.files.get(name=cache_file_name)
            cur_size_in_bytes += f.size_bytes
            size_dict[fn] = f.size_bytes
        if cur_size_in_bytes > flush_size_in_bytes:
            eval_logger.info(f"Flushing files cache. Current size: {cur_size_in_bytes} bytes.")
            for fn in sorted(list(size_dict.keys()), key=lambda x: size_dict[x], reverse=True):
                cache_file_name = self._files_cache[fn]
                eval_logger.info(f"Deleting file: {cache_file_name} of size {size_dict[fn]} bytes.")
                # import ipdb; ipdb.set_trace()
                self.client.files.delete(name=cache_file_name)
                self._files_cache.pop(fn, None)

    def clear_files_cache(self):
        """Clear the files cache by deleting all files uploaded to the client."""
        if self.use_vertexai:
            return
        eval_logger.info("Clearing files cache.")
        for f in self.client.files.list():
            eval_logger.info(f"Deleting file: {f.name} of size {f.size_bytes} bytes.")
            self.client.files.delete(name=f.name)
    
    def create_prompt(
        self,
        conversation: Dict[str, Any],
        **kwargs
    ):
        """Create a prompt for the Gemini model based on the conversation.

        Args:
            conversation (Dict[str, Any]): The conversation history.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The formatted prompt.
        """
        message_container = {"messages": list()}
        messages = list()
        
        for msg_idx in range(len(conversation)):
            
            input_msg = None
            generate_content_cfg = None
            
            if conversation[msg_idx]["tag"] == "task_instruction":
                generate_content_cfg = types.GenerateContentConfig(
                    system_instruction=conversation[msg_idx]["content"],
                    response_mime_type="application/json",
                    response_schema=self.response_schema,
                    temperature=self.generate_config.get("temperature", 0.0),
                    seed=self.generate_config.get("seed", 42),
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=self.thinking_config.get("thinking_budget", -1),
                        include_thoughts=self.thinking_config.get("include_thoughts", False),
                    ),
                    media_resolution=self.generate_config.get("media_resolution", 
                                                            types.MediaResolution.MEDIA_RESOLUTION_HIGH),
                )
                message_container["generate_content_config"] = generate_content_cfg
                
                
            elif conversation[msg_idx]["type"] == "video":                
                
                num_init_frames_to_preserve = None
                if conversation[msg_idx]["tag"].startswith("concat") and \
                    self.model_version != "gemini-2.5-flash":
                    num_init_frames_to_preserve = 1
                    if "tracking" in conversation[msg_idx]["tag"]:
                        num_init_frames_to_preserve = 2

                # NOTE: do repeated uploads of the same video file just to be sure
                video_file = None
                if not self.use_vertexai:
                    
                    video_fn = conversation[msg_idx]['content']
                    os.makedirs(self.tmpdir, exist_ok=True)
                    video_fn = process_video_if_needed(
                        video_path=video_fn,
                        max_duration=self.max_duration_of_videos,
                        tmpdir=self.tmpdir,
                        target_fps=self.video_fps,
                        num_init_frames_to_preserve=num_init_frames_to_preserve
                    )
                    
                    if video_fn not in self._files_cache:

                        eval_logger.info(f"Uploading video file: {video_fn}")
                        video_file = self.client.files.upload(
                            file=video_fn,
                        )
                        # Wait for the file to finish processing
                        while video_file.state.name == 'PROCESSING':
                            # print('Waiting for video to be processed.')
                            eval_logger.info('Waiting for video to be processed.')
                            time.sleep(2)
                            video_file = self.client.files.get(name=video_file.name)
                        self._files_cache[video_fn] = video_file.name
                    else:
                        eval_logger.info(f"Using cached video file for: {conversation[msg_idx]['content']}")
                        video_file_name = self._files_cache[video_fn]
                        video_file = self.client.files.get(name=video_file_name)
                        # time.sleep(10)
                    # video_file_uri = video_file.uri
                    # self._files_cache[conversation[msg_idx]["content"]] = video_file.uri
                else:
                    dest_path = osp.join(
                        "media_cache",
                        conversation[msg_idx]["content"].split("/")[-2],
                        osp.basename(conversation[msg_idx]["content"])
                    )
                    video_file = upload_video_if_absent(
                        bucket_name=self.gs_bucket,
                        local_path=conversation[msg_idx]["content"],
                        dest_path=dest_path
                    )
                    video_file = types.Part.from_uri(
                        file_uri=video_file,
                        mime_type="video/mp4",
                    )
                    # import ipdb; ipdb.set_trace()
                # input_msg = video_file_uri
                input_msg = video_file
                messages.append(input_msg)
                
                
            elif conversation[msg_idx]["type"] == "image":
                img_file = None
                if not self.use_vertexai:
                    # NOTE: trying to avoid uploading images, partly to debug discrepancies
                    # in performance on contact questions
                    # with open(conversation[msg_idx]["content"], "rb") as f:
                    #     image_bytes = f.read()
                    
                    if conversation[msg_idx]["content"] not in self._files_cache:
                        eval_logger.info(f"Uploading image file: {conversation[msg_idx]['content']}")
                        img_file = self.client.files.upload(
                            file=conversation[msg_idx]["content"],
                            
                        )
                    
                        while img_file.state.name == 'PROCESSING':
                            # print('Waiting for image to be processed.')
                            eval_logger.info('Waiting for image to be processed.')
                            time.sleep(1)
                            img_file = self.client.files.get(name=img_file.name)
                        self._files_cache[conversation[msg_idx]["content"]] = img_file.name
                    else:
                        eval_logger.info(f"Using cached image file for: {conversation[msg_idx]['content']}")
                        img_file_name = self._files_cache[conversation[msg_idx]["content"]]
                        img_file = self.client.files.get(name=img_file_name)
                else:
                    dest_path = osp.join(
                        "media_cache",
                        conversation[msg_idx]["content"].split("/")[-2],
                        osp.basename(conversation[msg_idx]["content"])
                    )
                    img_file = upload_video_if_absent(
                        bucket_name=self.gs_bucket,
                        local_path=conversation[msg_idx]["content"],
                        dest_path=dest_path,
                    )
                    ctype = mimetypes.guess_type(conversation[msg_idx]["content"])[0] 
                    if ctype is None or ctype == "image/jpg":
                        ctype = "image/jpeg"

                    img_file = types.Part.from_uri(
                        file_uri=img_file,
                        mime_type=ctype,
                    )

                input_msg = img_file
                messages.append(input_msg)
                
                
            elif conversation[msg_idx]["type"] == "text":
                input_msg = types.Part.from_text(
                    text=conversation[msg_idx]["content"],
                )
                messages.append(input_msg)
                
            else:
                raise ValueError(f"Unsupported message tag: {conversation[msg_idx]['tag']}")
        
        if "generate_content_config" not in message_container:
            generate_content_cfg = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=self.response_schema,
                temperature=self.generate_config.get("temperature", 0.0),
                seed=self.generate_config.get("seed", 42),
                thinking_config=types.ThinkingConfig(
                    thinking_budget=self.thinking_config.get("thinking_budget", -1),
                    include_thoughts=self.thinking_config.get("include_thoughts", False),
                ),
            )
            message_container["generate_content_config"] = generate_content_cfg
        message_container["messages"] = messages
        return message_container
    
    def forward(
        self, 
        conversation: Dict[str, Any],
        verbose: bool = False,
        **kwargs
    ):
        
        # import ipdb; ipdb.set_trace()
        
        messages = self.create_prompt(conversation, **kwargs)
        
        max_tries = MAX_TRIES
        thoughts = list()
        response = None
        # import ipdb; ipdb.set_trace()
        for attempt_idx in range(max_tries):
            try:
                # response = self.client.models.generate_content(
                #     model=self.model_version,
                #     contents=request["contents"],
                #     config=config,
                # )
                response = self.generate_content_with_backoff(
                    model=self.model_version,
                    contents=messages["messages"],
                    config=messages["generate_content_config"],
                )
                # import ipdb; ipdb.set_trace()
                if response.candidates:
                    for part in response.candidates[0].content.parts:
                        if not part.text:
                            continue
                        elif part.thought:
                            thoughts.append(part.text)

                break
            except Exception as e:
                eval_logger.error(f"Error in attempt {attempt_idx} for request: {e}")
                if attempt_idx == max_tries - 1:
                    eval_logger.error(f"Max tries reached for request. Skipping.")
                continue
        
        # NOTE: disabling this for now. I think we should be fine with size
        # self.check_files_cache()
        # self.clear_files_cache()
        response_text = response.text if response else ""
        if response_text is None:
            response_text = ""
        return {
            "response": response_text,
            "thoughts": thoughts
        }
            

    def post_process_response(self, response: Dict[str, Any]) -> str:
        """
        Post-process the response from the model.
        This can include cleaning up the response, formatting, etc.
        """
        return post_process_response(response)
        # if isinstance(response["response"], dict):
        #     return response["response"].get("answer", "").strip()
        # elif isinstance(response["response"], str):
        #     # import ipdb; ipdb.set_trace()
        #     match = re.search(r'\{.*\}', response["response"], re.DOTALL)
        #     filtered_response = response["response"][match.start():match.end()] if match else response["response"]
        #     try:
        #         response_dict = json.loads(filtered_response)
        #         return response_dict.get("answer", "").strip()
        #     except json.JSONDecodeError:
        #         eval_logger.info(f"Failed to decode JSON response: {response}")
        #         return response["response"].strip()
        # else:
        #     raise ValueError(f"Unsupported response type: {type(response['response'])}")
        
def post_process_response(response: Dict[str, Any]):
    """Post-process the response to clean up the string,
    return only the answer option

    Args:
        response (Dict[str, Any]): response dict containing the 
            raw response of the model under the "response" key
    """
    if isinstance(response, dict) and isinstance(response["response"], dict):
        return response["response"].get("answer", "").strip()
    elif isinstance(response, dict) and isinstance(response["response"], str):
        # import ipdb; ipdb.set_trace()
        match = re.search(r'\{.*\}', response["response"], re.DOTALL)
        filtered_response = response["response"][match.start():match.end()] if match else None
        # pattern = re.compile(r'^(?:[`]*json\s*)?\{(\\n)?\s*\"answer\"\s*:\s*[\"\']*([A-Z])[\"\']*\s*(\\n)?\}*|^[\"\']*([A-Z])[\"\']*$', re.MULTILINE)
        # match = pattern.search(response["response"])
        # if match:
        #     return match.group(1) or match.group(2)
        # else:
        #     return ""
        if filtered_response is not None:
            try:
                response_dict = json.loads(filtered_response)
                return response_dict.get("answer", "").strip()
            except json.JSONDecodeError:
                eval_logger.info(f"Failed to decode JSON response: {response}")
                return response["response"].strip()
        else:
            return response["response"].strip()
    else:
        return response
        
        
    
    
        
def main(
    conv_fn: str,
    model_name: str,
    api_key: str = None,
    thinking_config: Dict[str, Any] = {"thinking_budget": -1, "include_thoughts": True},
    generate_config: Dict[str, Any] = {"temperature": 0.0, "media_resolution": "high", "seed": 42},
    use_vertexai: bool = False,
    video_fps: int = 1,
    **kwargs
):
    import yaml
    
    model = Gemini(
        model_version=model_name,
        api_key=api_key,
        thinking_config=thinking_config,
        generate_config=generate_config,
        use_vertexai=use_vertexai,
        video_fps=video_fps,
    )
    
    with open(conv_fn, 'r') as f:
        conversation = yaml.safe_load(f)
    
    output = model.forward(conversation)
    print("Response:", output["response"])
    print("Thoughts:", output["thoughts"])
    
    post_processed_response = model.post_process_response(output)
    print("Post-processed response:", post_processed_response)
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gemini Model Evaluation")
    parser.add_argument(
        "--conv_fn",
        type=str,
        default=str(root / "src/eval/models/dummy_data/conversation_gemini.yaml"),
        help="Path to the conversation file in YAML format.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-2.5-flash",
        choices=["gemini-2.5-flash", "gemini-2.5-pro"],
        help="Model version to use for evaluation.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for authentication.",
    )
    
    args = parser.parse_args()
    
    main(
        conv_fn=args.conv_fn,
        model_name=args.model_name,
        api_key=args.api_key,
    )
