# `data` 📦

<a id="top"></a>

This directory contains the lightweight annotation snapshot used by the Flat-Pack Bench code release. The large media assets live in the official Hugging Face release:

- 🤗 [Flat-Pack Bench collection](https://huggingface.co/collections/justachetan/flat-pack-bench)
- 🧩 [Core dataset](https://huggingface.co/datasets/justachetan/flat-pack-bench)
- 🔀 [Additional analysis data](https://huggingface.co/datasets/justachetan/flat-pack-bench-misc)

Flat-Pack Bench builds on the IKEA Manuals at Work data and paper. Furniture categories, furniture names, video IDs, and the alignment between assembly videos and furniture instances follow that source dataset. This repository adds the benchmark-facing question set, prompt-time part masks, furniture part metadata, and part-ID scrambling variants used for evaluation.

<a id="table-of-contents"></a>

## 📚 Table of Contents

- [🗂️ Directory Map](#directory-map)
- [🧩 Questions JSONL](#questions-jsonl)
  - [Top-Level Fields](#questions-jsonl-top-level-fields)
  - [Nested `question` Object](#questions-jsonl-nested-question-object)
- [📝 Source Question YAMLs](#source-question-yamls)
- [🎭 Segmentation Masks](#segmentation-masks)
- [🪑 Furniture Part Annotations](#furniture-part-annotations)
  - [Per-Furniture Schema](#per-furniture-schema)
- [🔀 Scrambled Questions](#scrambled-questions)
  - [JSONL Files](#scrambled-jsonl-files)
  - [YAML Files](#scrambled-yaml-files)
- [🎭 Scrambled Segmentation Masks](#scrambled-segmentation-masks)
- [🧭 Usage Notes](#usage-notes)

<a id="directory-map"></a>

## 🗂️ Directory Map [⤴](#table-of-contents)

```text
data/
├── questions/
│   ├── questions.jsonl
│   └── yamls/
├── segmentation-masks/
├── furniture-annotations/
│   └── part-annotations/
├── scrambled-questions/
│   ├── jsonl/
│   └── yaml/
└── scrambled-segmentation-masks/
```

`questions/`, `segmentation-masks/`, and `furniture-annotations/` correspond to the core benchmark release. `scrambled-questions/` and `scrambled-segmentation-masks/` contain deterministic part-label permutations used to test whether models rely on stable part ID priors.

<a id="questions-jsonl"></a>

## 🧩 Questions JSONL [⤴](#table-of-contents)

Path: `questions/questions.jsonl`

This is the canonical benchmark table. Each line is a standalone JSON object for one multiple-choice question. The file has 602 rows.

<a id="questions-jsonl-top-level-fields"></a>

### Top-Level Fields [⤴](#table-of-contents)

| Field | Type | Meaning |
|---|---|---|
| `qid` | string | Stable hash identifier for the question contents. |
| `qid_flat` | string | Readable hierarchical ID combining question family, template, furniture/video identity, YAML ID, and frame/task index. |
| `question_category` | string | Broad skill family: `temporal_loc`, `temporal_ord`, `mating`, or `tracking`. |
| `template_type` | string | More specific template name, such as `track_single`, `track_multi`, `many_part_order`, `find_edges`, or `latest_change`. |
| `template_idx` | integer | Numeric template index used by the generation pipeline. |
| `vid_category` | string | IKEA Manuals at Work furniture category, such as `Chair`, `Table`, `Bench`, `Shelf`, `Desk`, or `Misc`. |
| `furniture_name` | string | Furniture instance name, for example `ronninge`, `stig`, `gladom`, or `laiva`. |
| `video_id` | string | Assembly video identifier inherited from the IKEA Manuals at Work alignment. |
| `frame_idx` | integer or list | Keyframe index for single-frame prompts, or a list of keyframes for tracking/correspondence prompts. |
| `question` | object | Rendered question text, answer options, and answer metadata. |
| `prompt_img_fn` | string, optional | Prompt image filename for single-image question variants. |
| `prompt_img_0_fn` | string, optional | First prompt image for tracking/correspondence questions. |
| `jumbled_prompt_img_1_fn` | string, optional | Second prompt image for tracking/correspondence questions where labels are rearranged. |
| `media_dir` | string, optional | Generation/cache provenance path. It is not required to interpret the benchmark item. |
| `video` | string, optional | Generated prompt video filename used by some evaluation pipelines. |

<a id="questions-jsonl-nested-question-object"></a>

### Nested `question` Object [⤴](#table-of-contents)

| Field | Type | Meaning |
|---|---|---|
| `raw_qstr` | string | Question text before answer choices are appended. |
| `qstr` | string | Full prompt text, including labeled answer choices. |
| `options` | object | Mapping from zero-based option index to an option record. |
| `correct_option` | object | Correct answer record with raw part ID/value, label, text, index, and full option text when available. |
| `num_options` | integer | Number of answer choices. |

Each option record usually has `raw`, `label`, `text`, and `full_text`. `raw` is the underlying answer value, while `label` and `full_text` describe the rendered multiple-choice option shown to models.

<a id="source-question-yamls"></a>

## 📝 Source Question YAMLs [⤴](#table-of-contents)

Path pattern: `questions/yamls/<question_id>.yaml`

These YAML files are the compact source form for the benchmark questions. They are easier to inspect by hand and are useful when code needs question parameters rather than fully rendered prompt strings.

| Field | Type | Meaning |
|---|---|---|
| `category` | string | Furniture category from IKEA Manuals at Work. |
| `name` | string | Furniture instance name. This corresponds to `furniture_name` in the JSONL. |
| `video_id` | string | Assembly video identifier. |
| `frame_idx` | integer or list | Keyframe or keyframes used by the prompt. |
| `question_category` | string | Broad benchmark skill family. |
| `template_type` | string | Template subtype used to generate the item. |
| `question_params` | object | Template-specific parameters, often including part IDs such as `query_part`. |
| `options` | list | Candidate answers. Each entry stores at least a `raw` value. |
| `correct_option` | object | Correct answer, including `raw` and zero-based `idx`. |

The YAMLs intentionally keep part IDs as strings. That matches the furniture annotation files and avoids accidental numeric coercion when IDs are used as object keys.

<a id="segmentation-masks"></a>

## 🎭 Segmentation Masks [⤴](#table-of-contents)

Path pattern:

```text
segmentation-masks/<category>/<furniture_name>/<video_id>/<video_id>.json
```

These files contain part-level masks for prompt frames. A mask file is nested by annotation source, keyframe index, and part ID:

```text
manual -> frame_index -> part_id -> mask_rle
```

Example shape:

```json
{
  "manual": {
    "0": {
      "3": {
        "size": [720, 1280],
        "counts": "..."
      }
    }
  }
}
```

| Level | Type | Meaning |
|---|---|---|
| `manual` | object | Top-level annotation source. |
| `manual.<frame_index>` | object | Masks for a keyframe, with the frame index stored as a string. |
| `manual.<frame_index>.<part_id>` | object | Mask for a furniture part. Part IDs match `furniture-annotations/`. |
| `size` | list of integers | Mask dimensions as `[height, width]`. |
| `counts` | string | Compressed COCO-style run-length encoding for the binary mask. |

The mask JSON stores geometry only. The semantic name for each part ID comes from the matching furniture annotation file.

<a id="furniture-part-annotations"></a>

## 🪑 Furniture Part Annotations [⤴](#table-of-contents)

Path patterns:

```text
furniture-annotations/part-annotations/<category>/<furniture_name>.json
furniture-annotations/part-annotations/index.json
```

The per-furniture files describe the physical part vocabulary used by questions and masks. `index.json` is a convenience aggregate keyed by `<category>/<furniture_name>` with the same records collected in one file.

<a id="per-furniture-schema"></a>

### Per-Furniture Schema [⤴](#table-of-contents)

| Field | Type | Meaning |
|---|---|---|
| `furniture_id` | string | `<category>/<furniture_name>` identifier. |
| `annotated_semantics` | object | Mapping from part ID string to human-readable part name. |
| `annotated_part_graph` | object | Undirected physical connectivity graph. Each part ID maps to the part IDs it touches or connects to in the assembled object. |
| `annotated_similar_parts` | object | Mapping from a part ID to visually or functionally similar part IDs. This is useful for reasoning about ambiguous repeated parts. |

Part IDs are strings such as `"0"`, `"1"`, and `"2"`. These IDs are the same labels shown in visual prompts and used in segmentation masks.

<a id="scrambled-questions"></a>

## 🔀 Scrambled Questions [⤴](#table-of-contents)

Paths:

```text
scrambled-questions/jsonl/questions_shuffled_part_ids*.jsonl
scrambled-questions/yaml/seed<seed>/<question_id>.yaml
```

Scrambled questions preserve the underlying videos, furniture, templates, and answers while changing the visible part labels. Each `seed<seed>` directory corresponds to one deterministic part-ID permutation.

<a id="scrambled-jsonl-files"></a>

### JSONL Files [⤴](#table-of-contents)

The JSONL files follow the same top-level schema as `questions/questions.jsonl`, but the rendered text and answer choices use shuffled part IDs. These files are convenient for batch evaluation.

<a id="scrambled-yaml-files"></a>

### YAML Files [⤴](#table-of-contents)

The YAML files follow the source question schema above and add:

| Field | Type | Meaning |
|---|---|---|
| `metadata.part_id_mapping` | object | Mapping from original part ID to shuffled display ID for the seed. |

Use `metadata.part_id_mapping` when you need to translate a scrambled label back to the original furniture annotation.

<a id="scrambled-segmentation-masks"></a>

## 🎭 Scrambled Segmentation Masks [⤴](#table-of-contents)

Path pattern:

```text
scrambled-segmentation-masks/seed<seed>/<category>/<furniture_name>/<video_id>/<video_id>.json
```

These masks have the same nested structure as `segmentation-masks/`, but the part IDs have already been remapped to match the labels shown in the corresponding scrambled question seed. Evaluation code should pair `scrambled-questions/yaml/seed<seed>/` with `scrambled-segmentation-masks/seed<seed>/` when operating in a shuffled-label setting.

<a id="usage-notes"></a>

## 🧭 Usage Notes [⤴](#table-of-contents)

- The annotations here are small enough to version with the code release. Full videos, frames, and larger media assets should be pulled from the Hugging Face data release.
- When joining files, use `(vid_category or category, furniture_name or name, video_id)` as the stable furniture/video identity.
- For model-facing question text, prefer `questions/questions.jsonl`. For generation or debugging, inspect the YAML source files.
- For part names and connectivity, load the matching `furniture-annotations/part-annotations/<category>/<furniture_name>.json` file.
- For mask rendering, decode `counts` and `size` with a COCO RLE-compatible utility such as `pycocotools.mask.decode`.
