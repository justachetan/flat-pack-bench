<a id="top"></a>

# `scripts`

Run all commands from the repository root. These launchers wrap
`src/eval/inference.py`, choose Hydra configs for common Flat-Pack Bench
evaluations, allocate GPUs across jobs, and write logs/results under ignored
`src/eval/` output directories.

<a id="table-of-contents"></a>

## Table of Contents

- [Scripts](#scripts)
- [Common Options](#common-options)
- [Changing Models](#changing-models)
- [Job Keys](#job-keys)

<a id="scripts"></a>

## Scripts [⤴](#table-of-contents)

| Script | Purpose |
|---|---|
| `run_model.sh` | Main benchmark prompt/video settings: mixed-media, concat, and collage on key-frame and trimmed videos. |
| `run_image_only_prompt.sh` | Image-only prompt ablations with and without HCVP-style visual prompting. |
| `run_contact_reasoning.sh` | Contact-reasoning ablations for old/new question templates and instructions. |
| `run_cot.sh` | Chain-of-thought prompt runs with deterministic and sampled seeds. |

All paths below are relative to the repo root by default:

| Input or output | Default |
|---|---|
| Data root | `data` |
| Key-frame videos | `data/videos/keyframe-video/fps-1` |
| Trimmed videos | `data/videos/trimmed-videos` |
| RGB frames | `data/rgb-frames` |
| Segmentation masks | `data/segmentation-masks` |
| Main question YAMLs | `data/questions/yamls` |
| Main question JSONL | `data/questions/questions.jsonl` |
| Logs | `src/eval/logs/<script-name>/` |
| Results | `src/eval/results/<script-name>/` |

Download data with:

```bash
python data/download_data.py --full-data
```

<a id="common-options"></a>

## Common Options [⤴](#table-of-contents)

Every launcher supports:

| Option | Effect |
|---|---|
| `--total-gpus N` | Use `N` GPUs instead of auto-detecting with `nvidia-smi`. |
| `--gpu-ids 0,2,3` | Restrict scheduling to specific device IDs. |
| `--job-gpus job=4,...` | Override GPUs assigned to selected job keys. |
| `--job-min-gpus N` | Require at least `N` GPUs per job, or use `job=N` pairs. |
| `--python PATH` | Use one Python interpreter for all jobs, or `job=PATH` pairs. |
| `--allow-shared-gpu` | Launch even if `nvidia-smi` reports existing GPU processes. |
| `--jobs a,b` | Run only selected job keys. |
| `--model NAME` | Override the Hydra model config group, such as `qwen_2_5_vl` or `internvl3`. |
| `--model-name NAME` | Override `model.model_name`, usually the Hugging Face checkpoint ID. |
| `--hydra-override KEY=VALUE` | Add one Hydra override; may be repeated. |
| `--hydra-overrides A=B,C=D` | Add comma-separated Hydra overrides. |
| `--run-tag TAG` | Prefix output directories and logs with a model/run label. |
| `--log-dir PATH` | Change the log directory. |
| `--data-root PATH` | Change the data root while preserving the expected layout under it. |
| `--results-root PATH` | Change where media caches, rendered templates, and responses are written. |

Use `--hydra-override` for one-off inference settings, for example:

```bash
scripts/run_model.sh \
  --total-gpus 8 \
  --jobs sep_media_first_keyframe \
  --hydra-override only_render_templates=True
```

<a id="changing-models"></a>

## Changing Models [⤴](#table-of-contents)

The model is controlled by two Hydra overrides:

- `--model` selects the config group in `src/eval/configs/model/`.
- `--model-name` selects the checkpoint or API model passed into that wrapper.

Qwen2.5-VL-72B:

```bash
scripts/run_model.sh \
  --total-gpus 8 \
  --job-min-gpus 8 \
  --model qwen_2_5_vl \
  --model-name Qwen/Qwen2.5-VL-72B-Instruct \
  --run-tag qwen_2_5_vl_72b
```

InternVL3-78B:

```bash
scripts/run_model.sh \
  --total-gpus 8 \
  --job-min-gpus 8 \
  --model internvl3 \
  --model-name OpenGVLab/InternVL3-78B-hf \
  --run-tag internvl3_78b
```

The paper-highlighted main settings differ by model family: InternVL3-78B uses
concat/key-frame for the strongest main result, while Qwen2.5-VL-72B uses
mixed-media/trimmed for its strongest main result. Run those directly with:

```bash
scripts/run_model.sh \
  --total-gpus 8 \
  --job-min-gpus 8 \
  --model internvl3 \
  --model-name OpenGVLab/InternVL3-78B-hf \
  --run-tag internvl3_78b \
  --jobs concat_media_first_keyframe

scripts/run_model.sh \
  --total-gpus 8 \
  --job-min-gpus 8 \
  --model qwen_2_5_vl \
  --model-name Qwen/Qwen2.5-VL-72B-Instruct \
  --run-tag qwen_2_5_vl_72b \
  --jobs sep_media_first_trimmed
```

<a id="job-keys"></a>

## Job Keys [⤴](#table-of-contents)

`run_model.sh`:

```text
sep_media_first_keyframe
concat_media_first_keyframe
collage_media_first_keyframe
sep_media_first_trimmed
concat_media_first_trimmed
collage_media_first_trimmed
```

`run_image_only_prompt.sh`:

```text
image_only_job_a  # HCVP image-only prompt
image_only_job_b  # no-HCVP image-only prompt
```

`run_contact_reasoning.sh`:

```text
contact_reasoning_job_a  # old template, old instructions
contact_reasoning_job_b  # new template, new instructions
contact_reasoning_job_c  # old template, new instructions
contact_reasoning_job_d  # new template, old instructions
```

`run_contact_reasoning.sh` expects these optional ablation assets:

```text
data/contact-reasoning/old_contact_reasoning_questions/question_yamls/
data/contact-reasoning/contact_reasoning_questions/question_yamls/
data/contact-reasoning/old_contact_reasoning_questions_cvpr.jsonl
data/contact-reasoning/contact_reasoning_questions_cvpr.jsonl
```

`run_cot.sh`:

```text
sep_media_first_keyframe
concat_media_first_keyframe
```

By default, `run_cot.sh` chooses the paper-appropriate setting for the selected
model:

- Qwen2.5-VL-72B: `sep_media_first_keyframe` with
  `qwen_2_5_vl_sep_media_first_with_thoughts`.
- InternVL3-78B: `concat_media_first_keyframe` with
  `qwen_2_5_vl_concat_media_first_with_thoughts`.

Qwen COT:

```bash
scripts/run_cot.sh \
  --total-gpus 8 \
  --job-min-gpus 8 \
  --model qwen_2_5_vl \
  --model-name Qwen/Qwen2.5-VL-72B-Instruct \
  --run-tag qwen_2_5_vl_72b_cot
```

InternVL3 COT:

```bash
scripts/run_cot.sh \
  --total-gpus 8 \
  --job-min-gpus 8 \
  --model internvl3 \
  --model-name OpenGVLab/InternVL3-78B-hf \
  --run-tag internvl3_78b_concat_keyframe_cot
```

`run_cot.sh` launches one deterministic run (`seed=42`, beam size 1) and five
sampled runs (`101`, `202`, `303`, `404`, `505`). Add `--sequential-seeds` if
the sampled runs should wait until the deterministic run has finished.
