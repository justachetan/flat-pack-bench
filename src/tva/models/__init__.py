import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
import torch
from src.tva.models.sam2 import SAM2Wrapper
# TODO: add the other models later
from src.eval.models import (
    gemini,
    openai_gpt,
    qwen_2_5_vl_hf
)

# NOTE: add new models here along with preferred device
#       (e.g., "cuda", "auto" or "cpu")
model_classes = {
    "sam2": (SAM2Wrapper, "cuda"),
    "gemini": (gemini.Gemini, "cpu"),
    "openai_gpt": (openai_gpt.OpenAIGPT, "cpu"),
    "qwen25_vl": (qwen_2_5_vl_hf.Qwen2_5_VlHF, "auto"),
}

model_factory = dict()

def forward(model_name, *args, init_args=None, **kwargs):
    # TODO: would be better to have model params for instantitation from 
    #       a config, and a way to decide GPUs that accounts for ``auto''
    #       setting in VLMs
    num_gpus = torch.cuda.device_count()
    class_name, preferred_device = list(model_classes[model_name])
    assert model_name in model_classes, f"Model {model_name} not supported"
    if model_name not in model_factory:
        device = preferred_device if preferred_device != "cuda" \
            else f"{preferred_device}:{len(model_factory) % num_gpus}"
        model_factory[model_name] = class_name(device=device, **(init_args or {}))
    # import ipdb; ipdb.set_trace()
    outputs = model_factory[model_name].forward(*args, **kwargs)
    
    return outputs

