import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from typing import Literal, Dict, Any

import time
from PIL import Image

import textwrap

# first run `conda activate perception_models`
from core.args import dataclass_from_dict
from core.transforms.image_transform import get_image_transform
from core.transforms.video_transform import get_video_transform
from src.eval.models.model_utils.perceptionlm.generate import (
    PackedCausalTransformerGeneratorArgs, 
    PackedCausalTransformerGenerator, 
    load_consolidated_model_and_tokenizer
)

from src.eval.models.base_model import BaseModel
from src.eval.models.model_utils.video_subspl.subspl_concat_video import subspl_concat_video
# from src.eval.models.qwen_2_5_vl_hf import post_process_response

class PerceptionLMModel(BaseModel):
    def __init__(
        self,
        model_name: Literal[
            "facebook/Perception-LM-1B",
            "facebook/Perception-LM-3B",
            "facebook/Perception-LM-8B",
        ] = "facebook/Perception-LM-8B",
        device: str = "cuda",
        temperature: float = 0.0,
        top_p: float = None,
        top_k: float = None,
        verbose: bool = False,
        num_frames: int = 4,
        **kwargs,
    ):
        """PerceptionLM by MetaAI

        NOTE: `max_new_tokens` equivalent for PerceptionLM seems to be
        initialized from the config

        Currently this class is written assuming a pure video input, i.e.
        it does not support mixed-media prompting.

        Args:
            model_name (Literal[ &quot;facebook, optional): name of the model. Defaults to "facebook/Perception-LM-8B".
            device (str, optional): device. Defaults to "cuda".
            temperature (float, optional): temperature for generation. Defaults to 0.0.
            top_p (float, optional): top_p value. Defaults to None.
            top_k (float, optional): top_k value. Defaults to None.
            verbose (bool, optional): print debug messages. Defaults to False.
            num_frames (int, optional): number of frames to use from the video input. Defaults to 4.
        """
        # import ipdb; ipdb.set_trace()
        super().__init__(**kwargs)
        self.model, self.tokenizer, self.config = load_consolidated_model_and_tokenizer(model_name)
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.num_frames = num_frames
        self.verbose = verbose

        # Create generator
        self.gen_cfg = dataclass_from_dict(
            PackedCausalTransformerGeneratorArgs,
            {"temperature": temperature, "top_p": top_p, "top_k": top_k},
            strict=False,
        )
        self.generator = PackedCausalTransformerGenerator(self.gen_cfg, self.model, self.tokenizer)
        self._tmp_video_dir = kwargs.get("tmp_video_dir", None)

    def _generate(
        self,
        media_path: str,
        question: str,
        media_type: str = "image",
        number_of_frames: int = 4,
        number_of_tiles: int = 1,
        temperature: float = 0.0,
        top_p: float = None,
        top_k=None,
        verbose: bool = False,
    ):
        
        prompts = []
        if media_type == "image":
            transform = get_image_transform(
                vision_input_type=(
                    "vanilla" if number_of_tiles == 1 else self.config.data.vision_input_type
                ),
                image_res=self.model.vision_model.image_size,
                max_num_tiles=number_of_tiles,
            )
            image = Image.open(media_path).convert("RGB")
            image, _ = transform(image)
            prompts.append((question, image))
        elif media_type == "video":
            transform = get_video_transform(
                image_res=self.model.vision_model.image_size,
            )
            video_info = (media_path, number_of_frames, None, None, None)
            frames, _ = transform(video_info)
            prompts.append((question, frames))
        else:
            raise NotImplementedError(
                f"The provided generate function only supports image and video."
            )
        
        # Run generation
        start_time = time.time()
        generation, loglikelihood, greedy = self.generator.generate(prompts)
        end_time = time.time()
        if self.verbose or verbose:
            for i, gen in enumerate(generation):
                # Calculate tokens per second
                total_tokens = sum(
                    len(self.tokenizer.encode(gen, False, False)) for gen in generation
                )
                tokens_per_second = total_tokens / (end_time - start_time)
                print("=================================================")
                print(textwrap.fill(gen, width=75))
                print(f"Tokens per second: {tokens_per_second:.2f}")
                print("=================================================")
        
 
        return generation[0]
    
    def create_prompt(self, conversation):
        """
        NOTE: For PerceptionLM right now we only handle
        pure video prompts, i.e., concat or collage 
        visual prompts

        Args:
            conversation (List[Dict[str, str]]): Loaded conversation
        """
        question = ""
        video_fn = None
        for msg_idx, msg in enumerate(conversation):
            msg_type = msg["type"]
            content = msg["content"]
            tag = msg["tag"]

            
            # import ipdb; ipdb.set_trace()
            if msg_type == "video":
                video_fn = content
                if tag.startswith("concat"):
                    initial_preserve_indices = [0] # preserve the first frame as visual prompt for concat videos
                    if "tracking" in tag:
                        # for tracking videos, we preserve the first two frames to provide better visual prompt for tracking
                        initial_preserve_indices = [0, 1]
                    output_video_subspl_dir = self._tmp_video_dir if self._tmp_video_dir is not None else None
                    video_fn = subspl_concat_video(
                        video_fn,
                        output_dir=output_video_subspl_dir, # this way it will be saved in the same directory as the input video, i.e., the media cache
                        num_subspl_frames=self.num_frames,
                        initial_preserve_indices=initial_preserve_indices,
                    )
            
            elif msg_type == "text":
                question += content

        return question, video_fn
    
    def forward(
        self,
        conversation: Dict[str, Any],
        verbose: bool = False,
    ):

        question, media_path = self.create_prompt(conversation)
        answer = self._generate(
            media_path=media_path,
            question=question,
            media_type="video",
            number_of_frames=self.num_frames,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            verbose=verbose,
        )

        return answer

    def post_process_response(self, response):
        # return post_process_response(response)
        return response.strip().split(".")[0].strip()



def main(
    conversation: Dict[str, Any],
    model_name: str = "facebook/Perception-LM-8B",
    device: str = "cuda",
    temperature: float = 0.0,
    top_p: float = None,
    top_k: float = None,
    num_frames: int = 32,
    verbose: bool = True,
    tmp_video_dir: str = None,
):
    model = PerceptionLMModel(
        model_name=model_name,
        device=device,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_frames=num_frames,
        verbose=verbose,
        tmp_video_dir=tmp_video_dir,
    )

    

    answer = model.forward(conversation, verbose=verbose)
    print("Generated Answer:")
    print(answer)
    print("Post-processed Answer:")
    print(model.post_process_response(answer))

if __name__ == "__main__":
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        type=str,
        default="facebook/Perception-LM-1B",
        help="Name of the PerceptionLM model to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the model on.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p value for nucleus sampling.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k value for sampling.",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=32,
        help="Number of frames to generate.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    parser.add_argument(
        "--conv-fn",
        type=str,
        default=str(root / "src/eval/models/dummy_data/conversation.yaml"),
        help="Path to the conversation JSON file.",
    )
    parser.add_argument(
        "--tmp-video-dir",
        type=str,
        default=None,
        help="Temporary directory to store subsampled videos. \
            If not provided, subsampled videos will be stored \
            in the same directory as the input videos.",
    )

    args = parser.parse_args()
    
    with open(args.conv_fn, "r") as f:
       conversation = yaml.safe_load(f)

    main(
        conversation=conversation,
        model_name=args.model_name,
        device=args.device,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        num_frames=args.num_frames,
        verbose=args.verbose,
        tmp_video_dir=args.tmp_video_dir,
    )



    
