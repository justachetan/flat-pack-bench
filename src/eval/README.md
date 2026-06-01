<a id="top"></a>

# `src/eval`

Evaluation code for Flat-Pack Bench. This package renders benchmark questions
into model-specific conversations, builds the visual/video media used in those
conversations, runs model inference, and computes aggregate metrics.

Run commands below from the repository root.

<a id="table-of-contents"></a>

## Table of Contents

- [Directory Map](#directory-map)
- [Configs](#configs)
- [Models](#models)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Tabulate Results](#tabulate-results)
- [Common Outputs](#common-outputs)
- [Adding A Model Or Prompt Setting](#adding-a-model-or-prompt-setting)

<a id="directory-map"></a>

## Directory Map [⤴](#table-of-contents)

```text
src/eval/
├── configs/                  # Hydra configs for inference, models, and media pipelines
├── models/                   # Model wrappers and model-specific utilities
├── prompts/                  # Prompt/media rendering code and conversation templates
├── evaluate.py               # Accuracy computation for response JSONL files
├── inference.py              # Hydra entrypoint for rendering prompts and running inference
└── tabulate_results.py       # Result-table generation with bootstrap confidence intervals
```

<a id="configs"></a>

## Configs [⤴](#table-of-contents)

[configs/](configs/) contains the Hydra configuration used by
[inference.py](inference.py).

Top-level inference configs, such as
[base_inference.yaml](configs/base_inference.yaml) and
[qwen_2_5_vl_sep_media_first.yaml](configs/qwen_2_5_vl_sep_media_first.yaml),
combine three pieces:

| Field | Purpose |
|---|---|
| `defaults` | Selects the media pipeline config and model config. |
| `media_cache_dir` | Where generated videos and visual prompts are cached. |
| `rendered_template_dir` | Where rendered conversations and question metadata are written. |
| `responses_dir` | Where `responses.jsonl` and the resolved run config are written. |
| `video_dir`, `mask_dir`, `img_dir`, `question_dir` | Input data roots for videos, masks, RGB frames, and source question YAMLs. |
| `num_shuffles` | Number of answer-option shuffles to render per source question. |
| `override_cache` | List of question YAML filenames, such as `012.yaml`, to rerun. |
| `override_templates` | Rerender cached templates for entries listed in `override_cache`. |
| `only_render_templates` | Build media and conversations, then skip model inference. |

[configs/model/](configs/model/) defines model-specific Hydra targets. Each
file points `_target_` at a class in [models/](models/) and stores generation
parameters such as `model_name`, temperature, frame sampling, and token limits.

[configs/media_pipeline/](configs/media_pipeline/) defines prompt/media
recipes. Each pipeline config chooses a sequence of components from
[configs/media_pipeline/components/](configs/media_pipeline/components/) and a
conversation template key such as `SEP_MEDIA_FIRST_PROACTIVE_COMMON`,
`ONLY_IMAGE_FIRST`, `COLLAGE_VP_LEFT_MEDIA_FIRST_PROACTIVE_COMMON`, or
`CONCAT_VP_FIRST_MEDIA_FIRST_PROACTIVE_COMMON_WITH_THOUGHTS_V2`.

The media components include video loading, visual-prompt generation, separate
media attachment, collage rendering, visual-prompt/video concatenation,
subsampling, embedding, and optional frame numbering.

<a id="models"></a>

## Models [⤴](#table-of-contents)

[models/](models/) contains the model wrappers used by inference. Model classes
inherit from [BaseModel](models/base_model.py), whose expected interface is:

| Method | Role |
|---|---|
| `create_prompt(...)` | Convert a rendered conversation into the prompt format expected by the backend model. |
| `forward(...)` | Run model inference and return the raw response. |
| `post_process_response(...)` | Normalize the raw response into the answer label used for scoring. |

The directory includes wrappers for local and hosted VLMs such as Gemini,
OpenAI GPT models, Qwen VL variants, LLaVA variants, InternVL, Aria, ArrowRL,
GenFS, Perception-LM, and VideoRefer. The exact default checkpoint or API model
for each wrapper is configured in [configs/model/](configs/model/).

[models/model_utils/](models/model_utils/) stores helper code that is specific
to particular model families:

| Path | Purpose |
|---|---|
| [perceptionlm/](models/model_utils/perceptionlm/) | Perception-LM tokenizer, transformer/generation helpers, and response post-processing. |
| [video_subspl/subspl_concat_video.py](models/model_utils/video_subspl/subspl_concat_video.py) | Subsamples concatenated videos while preserving the initial visual-prompt frames. |
| [videorefer/convert_data_format.py](models/model_utils/videorefer/convert_data_format.py) | Converts Flat-Pack Bench questions and prompt masks into VideoRefer's prompt-mask JSONL format. See [PixelRefer VideoRefer](https://github.com/alibaba-damo-academy/PixelRefer/tree/main/VideoRefer) for the downstream format and model context. |
| [qwen25_vl_vllm_online.jinja](models/model_utils/qwen25_vl_vllm_online.jinja) | Chat template for Qwen2.5-VL vLLM-style serving. |

<a id="inference"></a>

## Inference [⤴](#table-of-contents)

[inference.py](inference.py) is the main entrypoint for running a model on the
benchmark. It:

1. Loads a Hydra config from [configs/](configs/).
2. Instantiates the selected model from `cfg.model`.
3. Builds media and rendered conversations for source YAMLs in `question_dir`.
4. Writes rendered templates and metadata into `rendered_template_dir`.
5. Runs each conversation through the model.
6. Appends raw and post-processed responses to `responses_dir/responses.jsonl`.

Example:

```bash
python src/eval/inference.py --config-name qwen_2_5_vl_sep_media_first
```

Useful Hydra overrides:

```bash
python src/eval/inference.py \
  --config-name qwen_2_5_vl_sep_media_first \
  model.model_name=Qwen/Qwen2.5-VL-32B-Instruct \
  responses_dir=src/eval/results/responses/my_run
```

Render prompts without calling the model:

```bash
python src/eval/inference.py \
  --config-name qwen_2_5_vl_sep_media_first \
  only_render_templates=True
```

Rerun selected questions:

```bash
python src/eval/inference.py \
  --config-name qwen_2_5_vl_sep_media_first \
  override_cache='[012.yaml,045.yaml]' \
  override_templates=True
```

<a id="evaluation"></a>

## Evaluation [⤴](#table-of-contents)

[evaluate.py](evaluate.py) scores one or more `responses.jsonl` files. It uses
the model key to select the same post-processing function used by the model
wrapper, compares the processed response to the correct option label, and
returns category-level metrics.

Example:

```bash
python src/eval/evaluate.py \
  --responses_fn src/eval/results/responses/qwen_2_5_vl_32b_sep_media_first/responses.jsonl \
  --model_name qwen_2_5_vl_hf
```

Multiple files can be passed with `--responses_fns`:

```bash
python src/eval/evaluate.py \
  --responses_fns run_a/responses.jsonl run_b/responses.jsonl \
  --model_name gemini
```

Reported metrics:

| Metric | Meaning |
|---|---|
| `strict` | A source question is correct only if all considered shuffles are correct. |
| `majority` | A source question is correct if more shuffles are correct than incorrect. |
| `any` | A source question is correct if at least one considered shuffle is correct. |
| `random_chance` | Mean chance accuracy implied by the number of answer options. |
| `freq_chance` | Accuracy of always choosing the most frequent correct label in a category. |

By default, the CLI considers one shuffle via `--num_shuffs_to_consider 1`.
Use `--binary_resp_acc` for yes/no-style binary response accuracy.

<a id="tabulate-results"></a>

## Tabulate Results [⤴](#table-of-contents)

[tabulate_results.py](tabulate_results.py) scans a directory of response
runs, evaluates each run, and computes video-level hierarchical bootstrap
confidence intervals for model accuracy.

Example:

```bash
python src/eval/tabulate_results.py \
  --responses-dir src/eval/results/responses \
  --output-csv src/eval/results/summary.csv
```

Each result subdirectory should usually contain:

```text
responses.jsonl
config.yaml
```

The script infers model name, model key, prompt setting, and number of shuffles
from `config.yaml`. VideoRefer result directories are handled specially by name.

<a id="common-outputs"></a>

## Common Outputs [⤴](#table-of-contents)

Inference writes three main output groups:

| Output | Contents |
|---|---|
| `media_cache_dir` | Generated videos, visual prompts, collages, concatenated media, and other reusable media artifacts. |
| `rendered_template_dir` | `conversation_metadata.json`, `question_metadata.json`, `questions.jsonl`, and one folder per rendered conversation containing `conversation.yaml` and `question.json`. |
| `responses_dir` | Resolved run config snapshots and `responses.jsonl`, where each line has `conv_id`, `question`, raw `response`, and `post_processed_response`. |

These directories are cache-oriented. Reusing them avoids rerendering media and
rerunning completed model calls unless `override_cache` and related flags are
set.

<a id="adding-a-model-or-prompt-setting"></a>

## Adding A Model Or Prompt Setting [⤴](#table-of-contents)

To add a model:

1. Implement a wrapper in [models/](models/) using the [BaseModel](models/base_model.py) interface.
2. Add a Hydra config in [configs/model/](configs/model/) with `_target_` set to the wrapper class.
3. Add the model's response parser to `post_process_registry` in [evaluate.py](evaluate.py).
4. Create or override a top-level inference config that selects the new model.

To add a prompt/media setting:

1. Add or reuse component configs in [configs/media_pipeline/components/](configs/media_pipeline/components/).
2. Add a pipeline recipe in [configs/media_pipeline/](configs/media_pipeline/).
3. Choose the matching `conv_template` from `ConvTemplateRegistry` in [prompts/builder.py](prompts/builder.py), or add a new conversation template under [prompts/templates/conversations/](prompts/templates/conversations/).
4. Create a top-level inference config that overrides `media_pipeline`.
