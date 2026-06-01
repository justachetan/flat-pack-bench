import hydra
from omegaconf import DictConfig
import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from typing import Tuple, Literal, List, Any, Dict, Union
import os
import yaml
import json
import os.path as osp

import torch

from tqdm import tqdm

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything

from src.eval.prompts.builder import TemplateBuilder
from src.eval.models.base_model import BaseModel


def run_inference(
    video_dir: str,
    mask_dir: str,
    img_dir: str,
    model: BaseModel,
    media_pipeline_cfg: DictConfig,
    media_cache_dir: str,
    question_dir: str,
    rendered_template_dir: str,
    responses_dir: str,
    question_jsonl_fn: str = None,
    num_shuffles: int = 1,
    override_cache: List[str] = [],
    override_templates: bool = False,
    only_render_templates: bool = False,
):
    template_builder = TemplateBuilder(
        video_dir,
        mask_dir,
        img_dir,
        media_pipeline_cfg,
        media_cache_dir,
        num_shuffles
    )
    
    # Load questions from the specified directory
    question_files = sorted(
        [f for f in os.listdir(question_dir) if f.endswith('.yaml')],
        key=lambda x: int(x.split(".")[0])
    )
    
    pbar = tqdm(question_files, total=len(question_files), 
                desc="Building templates", leave=False)
    for question_fn in pbar:
        pbar.set_description(f"Building template: {question_fn}")
        question_yaml_path = osp.join(question_dir, question_fn)
        
        # Build the template for the current question
        template_meta = template_builder.build(
            yaml_fn=question_yaml_path,
            rendered_template_dir=rendered_template_dir,
            override_cache=(question_fn in override_cache) and override_templates,
            question_jsonl_fn=question_jsonl_fn,
        )
        
    pbar.close()
    
    # TODO: put model inference here:
    #   should be capable of handling template files
    #   absorb data and return the response after any post-processing
    conv_meta_fn = osp.join(rendered_template_dir, "conversation_metadata.json")
    conv_meta = dict()
    if osp.exists(conv_meta_fn):
        with open(conv_meta_fn, 'r') as f:
            conv_meta = json.load(f)
    os.makedirs(responses_dir, exist_ok=True)
    
    # Check if responses already exist
    resp_cache = list()
    if osp.exists(osp.join(responses_dir, "responses.jsonl")):
        with open(osp.join(responses_dir, "responses.jsonl"), "r") as f:
            for line in f:
                resp_cache.append(json.loads(line.strip())["conv_id"])
    
    if only_render_templates:
        print("Only rendering templates, skipping inference.")
        return
    
          
    pbar = tqdm(enumerate(list(conv_meta.keys())), total=len(conv_meta),
                desc="Running inference", leave=False)
    for idx, conv_id in pbar:
        
        question_yaml_for_conv = conv_meta[conv_id]["qid_flat"].split("/")[-2] + ".yaml"
        if conv_id in resp_cache:
            
            if question_yaml_for_conv not in override_cache:
                pbar.set_description(f"Skipping inference: {conv_id[:10] + '...'}")
                pbar.update(1)
                continue
        
        pbar.set_description(f"Running inference: {conv_id[:10] + '...'}")
        conv_fn = osp.join(rendered_template_dir, conv_id, "conversation.yaml")
        question_fn = osp.join(rendered_template_dir, conv_id, "question.json")
        
        with open(question_fn, 'r') as f:
            question = json.load(f)
        
        with open(conv_fn, 'r') as f:
            conv = yaml.safe_load(f)
        # Run the model inference
        with torch.no_grad():
            response = model(conv)
        post_process_response = model.post_process_response(response)
        
        if override_cache is None or len(override_cache) == 0 \
            or (question_yaml_for_conv not in override_cache):
            
            with open(osp.join(responses_dir, "responses.jsonl"), "a+") as f:
                f.write(json.dumps({
                    "conv_id": conv_id,
                    "question": question,
                    "response": response,
                    "post_processed_response": post_process_response
                }) + "\n")
        else:
            # import ipdb; ipdb.set_trace()
            with open(osp.join(responses_dir, f"responses.jsonl")) as f:
                resps_cache = [json.loads(line.strip()) for line in f]
            for resp_idx in range(len(resps_cache)):
                resp = resps_cache[resp_idx]
                question_yaml_for_cache_conv = resp["question"]["qid_flat"].split("/")[-2] + ".yaml"
                if question_yaml_for_cache_conv in override_cache and \
                    question_yaml_for_cache_conv == question_yaml_for_conv:
                    # assert resp["conv_id"] == conv_id, \
                    #     f"Conversation ID {resp['conv_id']} already exists in responses.jsonl, " \
                    #     f"but does not match the overriding conv_id {conv_id}. Usually this means " \
                    #         f"that the question YAML has changed significantly (type change, etc.) "
                    
                    resps_cache[resp_idx] = {
                        "conv_id": conv_id,
                        "question": question,
                        "response": response,
                        "post_processed_response": post_process_response
                    }
                    break
                
            with open(osp.join(responses_dir, "responses.jsonl"), "w") as f:
                for resp in resps_cache:
                    f.write(json.dumps(resp) + "\n")
                        


    pbar.close()
    

# Hydra entrypoint for running inference
# @hydra.main(config_path="configs", config_name="qwen_2_5_vl_sep_media_first", version_base="1.1")
@hydra.main(config_path="configs", config_name="base_inference", version_base="1.1")
def main(cfg: DictConfig):
    """
    Hydra entrypoint for running inference.
    Expects a config file named 'qwen_2_5_vl_sep_media_first.yaml'
    under src/eval/configs/media/.
    """
    # Optional: print full config
    from omegaconf import OmegaConf
    print(OmegaConf.to_yaml(cfg, resolve=True))

    os.makedirs(cfg.responses_dir, exist_ok=True)
    cfg_dump_fn = osp.join(cfg.responses_dir, "config.yaml")
    if osp.exists(cfg_dump_fn):
        existing_cfg_files = [f for f in os.listdir(cfg.responses_dir) if f.endswith('.yaml')]
        cfg_dump_fn = osp.join(cfg.responses_dir, f"config_{len(existing_cfg_files)}.yaml")
        
    with open(cfg_dump_fn, 'w') as f:
        yaml.dump(OmegaConf.to_container(cfg, resolve=True), f)
        
    seed = cfg.seed if "seed" in cfg else 42
    print(f"Setting seed to: {seed}")
    seed_everything(seed)
        
    model = hydra.utils.instantiate(cfg.model)
    
    run_inference(
        video_dir=cfg.video_dir,
        mask_dir=cfg.mask_dir,
        img_dir=cfg.img_dir,
        model=model,
        media_pipeline_cfg=cfg.media_pipeline,
        media_cache_dir=cfg.media_cache_dir,
        question_dir=cfg.question_dir,
        rendered_template_dir=cfg.rendered_template_dir,
        responses_dir=cfg.responses_dir,
        question_jsonl_fn=cfg.get("question_jsonl_fn", None),
        num_shuffles=cfg.get("num_shuffles", 0),
        override_cache=cfg.get("override_cache", []),
        override_templates=cfg.get("override_templates", False),
        only_render_templates=cfg.get("only_render_templates", False)
    )


if __name__ == "__main__":
    # python3 inference.py --config-name qwen_2_5_vl_sep_media_first
    main()
    
