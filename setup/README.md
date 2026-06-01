# Flat-Pack Bench Setup

Run commands from the repository root unless a step explicitly changes into an external checkout. These environment files are normalized snapshots for reproducing the benchmark and model-specific baselines.

## Default FPB Environment

Use `fpb` for data downloads, prompt rendering, evaluation, and common model wrappers.

```bash
conda env create -f setup/fpb.yml
conda activate fpb
python -m pip install -r setup/requirements_benchmark.txt

export FPB_ROOT="$(pwd)"
export PYTHONPATH="${FPB_ROOT}:${FPB_ROOT}/src:${PYTHONPATH}"
```

## LLaVA-NeXT Environment

Use `fpb-llava` for LLaVA-NeXT models from [LLaVA-VL/LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT).

```bash
conda env create -f setup/fpb-llava.yml
conda activate fpb-llava

mkdir -p external
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git external/LLaVA-NeXT
python -m pip install -e "external/LLaVA-NeXT[train]"

export FPB_ROOT="$(pwd)"
export LLAVA_NEXT_ROOT="${FPB_ROOT}/external/LLaVA-NeXT"
export PYTHONPATH="${FPB_ROOT}:${FPB_ROOT}/src:${LLAVA_NEXT_ROOT}:${PYTHONPATH}"
```

If you keep LLaVA-NeXT outside this repository, set `LLAVA_NEXT_ROOT` to that checkout instead.

## PerceptionLM Environment

Use `fpb-plm` for PerceptionLM models from [facebookresearch/perception_models](https://github.com/facebookresearch/perception_models).

```bash
conda env create -f setup/fpb-plm.yml
conda activate fpb-plm

mkdir -p external
git clone https://github.com/facebookresearch/perception_models.git external/perception_models
python -m pip install -e external/perception_models

export FPB_ROOT="$(pwd)"
export PERCEPTION_MODELS_ROOT="${FPB_ROOT}/external/perception_models"
export PYTHONPATH="${FPB_ROOT}:${FPB_ROOT}/src:${PERCEPTION_MODELS_ROOT}:${PYTHONPATH}"
```

If you keep `perception_models` outside this repository, set `PERCEPTION_MODELS_ROOT` to that checkout instead.

## SAM2 Environment For TVA

Use `fpb-sam2` for Temporal Video Agent runs that need fresh SAM2 video-object segmentations. This environment starts from the default FPB environment, so install the default FPB requirements as part of setup.

```bash
conda env create -f setup/fpb-sam2.yml
conda activate fpb-sam2
python -m pip install -r setup/requirements_benchmark.txt
```

Install PyTorch and torchvision for your CUDA platform before installing SAM2. Follow the official [SAM2 installation guide](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md) and choose the PyTorch command that matches your driver and CUDA version.

Then install SAM2 from [facebookresearch/sam2](https://github.com/facebookresearch/sam2):

```bash
mkdir -p external
git clone https://github.com/facebookresearch/sam2.git external/sam2
python -m pip install -e "external/sam2[notebooks]"

export FPB_ROOT="$(pwd)"
export SAM2_ROOT="${FPB_ROOT}/external/sam2"
export PYTHONPATH="${FPB_ROOT}:${FPB_ROOT}/src:${SAM2_ROOT}:${PYTHONPATH}"
```

If the SAM2 CUDA extension is not needed or fails to build on your machine, use the upstream fallback:

```bash
SAM2_BUILD_CUDA=0 python -m pip install -e "external/sam2[notebooks]"
```

TVA's SAM2 wrapper defaults to the SAM2.1 large checkpoint at:

```text
src/tva/models/ckpts/sam2.1_hiera_large.pt
```

Download the SAM2.1 checkpoints from the SAM2 repository and place or symlink `sam2.1_hiera_large.pt` at that path before running TVA segmentations.

## Notes

- `fpb.yml` and `fpb-sam2.yml` are normalized from the default FPB environment exported in `vlm-4d-bench`; `requirements_benchmark.txt` is copied from the same source.
- The setup YAMLs remove machine-specific `prefix:` entries.
- The environment names are `fpb`, `fpb-llava`, `fpb-plm`, and `fpb-sam2`.
- Large datasets, checkpoints, and generated outputs should stay outside git-tracked files.
