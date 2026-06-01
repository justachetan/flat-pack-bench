#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

readonly -a JOB_KEYS=(
    "sep_media_first_keyframe"
    "concat_media_first_keyframe"
    "collage_media_first_keyframe"
    "sep_media_first_trimmed"
    "concat_media_first_trimmed"
    "collage_media_first_trimmed"
)

print_usage() {
    cat <<'EOF'
Usage: scripts/run_model.sh [OPTIONS]

Run Flat-Pack Bench model inference across the main prompt/video settings.

Optional arguments:
  --total-gpus N         Override the total number of GPUs visible to this script.
  --gpu-ids SPEC         Comma separated GPU device IDs to schedule.
  --job-gpus SPEC        Comma separated overrides such as job=gpus,job=gpus.
  --job-min-gpus SPEC    Minimum GPUs per job, either a single integer or job=min pairs.
  --python SPEC          Interpreter path, or comma separated job=/path/to/python pairs.
  --allow-shared-gpu     Skip waiting for GPUs to go idle before launching jobs.
  --jobs SPEC            Comma separated list of job keys to run.
  --model NAME           Hydra model config group to select (default: qwen_2_5_vl).
  --model-name STR       Value forwarded to Hydra as model.model_name.
  --hydra-override STR   Additional Hydra override to forward; may be repeated.
  --hydra-overrides STR  Comma separated Hydra overrides to forward.
  --run-tag TAG          Optional suffix/prefix for outputs and logs.
  --log-dir PATH         Directory for per-job logs.
  --data-root PATH       Benchmark data root (default: data).
  --results-root PATH    Output root (default: src/eval/results/run_model).
  -h, --help             Show this message.

Available job keys:
EOF
    printf '  %s\n' "${JOB_KEYS[@]}"
}

log() {
    printf '[%s] %s\n' "$(date +%T)" "$*" >&2
}

die() {
    log "ERROR: $*"
    exit 1
}

hydra_quote() {
    local value="$1"
    value=${value//\'/\'\\\'\'}
    printf "'%s'" "$value"
}

resolve_path() {
    local base="$1"
    local suffix="$2"
    base="${base%/}"
    suffix="${suffix#/}"
    echo "${base}/${suffix}"
}

format_response_dir_name() {
    local base="$1"
    if [[ -n "$RUN_TAG" ]]; then
        echo "${RUN_TAG}_${base}"
    else
        echo "$base"
    fi
}

build_common_hydra_overrides() {
    COMMON_HYDRA_OVERRIDES=()
    if [[ -n "$MODEL" ]]; then
        COMMON_HYDRA_OVERRIDES+=("model=$MODEL")
    fi
    if [[ -n "$MODEL_NAME" ]]; then
        COMMON_HYDRA_OVERRIDES+=("model.model_name=$(hydra_quote "$MODEL_NAME")")
    fi
    if (( ${#USER_HYDRA_OVERRIDES[@]} > 0 )); then
        COMMON_HYDRA_OVERRIDES+=("${USER_HYDRA_OVERRIDES[@]}")
    fi
}

is_valid_job_key() {
    local candidate="$1"
    local key
    for key in "${JOB_KEYS[@]}"; do
        if [[ "$candidate" == "$key" ]]; then
            return 0
        fi
    done
    return 1
}

parse_job_gpus_arg() {
    local spec="$1"
    local pair key value
    IFS=',' read -r -a pairs <<< "$spec"
    unset IFS
    for pair in "${pairs[@]}"; do
        [[ -z "$pair" ]] && continue
        [[ "$pair" == *=* ]] || die "Invalid --job-gpus entry '$pair' (expected job=gpus)."
        key="${pair%%=*}"
        value="${pair#*=}"
        is_valid_job_key "$key" || die "Unknown job key '$key' in --job-gpus."
        [[ "$value" =~ ^[0-9]+$ ]] || die "GPU count for '$key' must be a positive integer."
        JOB_GPU_OVERRIDES["$key"]="$value"
    done
}

parse_job_min_gpus_arg() {
    local spec="$1"
    local pair key value
    if [[ "$spec" =~ ^[0-9]+$ ]]; then
        for key in "${JOB_KEYS[@]}"; do
            JOB_MIN_GPU_OVERRIDES["$key"]="$spec"
        done
        return
    fi
    IFS=',' read -r -a pairs <<< "$spec"
    unset IFS
    for pair in "${pairs[@]}"; do
        [[ -z "$pair" ]] && continue
        [[ "$pair" == *=* ]] || die "Invalid --job-min-gpus entry '$pair' (expected job=min)."
        key="${pair%%=*}"
        value="${pair#*=}"
        is_valid_job_key "$key" || die "Unknown job key '$key' in --job-min-gpus."
        [[ "$value" =~ ^[0-9]+$ ]] || die "Minimum GPU count for '$key' must be a positive integer."
        JOB_MIN_GPU_OVERRIDES["$key"]="$value"
    done
}

parse_job_python_arg() {
    local spec="$1"
    local pair key value
    IFS=',' read -r -a pairs <<< "$spec"
    unset IFS
    for pair in "${pairs[@]}"; do
        [[ -z "$pair" ]] && continue
        [[ "$pair" == *=* ]] || die "Invalid --python entry '$pair' (expected job=/path/to/python)."
        key="${pair%%=*}"
        value="${pair#*=}"
        is_valid_job_key "$key" || die "Unknown job key '$key' in --python."
        [[ -n "$value" ]] || die "Python interpreter for '$key' cannot be empty."
        JOB_PYTHON_OVERRIDES["$key"]="$value"
    done
}

parse_selected_jobs_arg() {
    local spec="$1"
    JOB_FILTER_ENABLED=1
    SELECTED_JOB_KEYS=()
    JOB_FILTER_SET=()
    IFS=',' read -r -a SELECTED_JOB_KEYS <<< "$spec"
    unset IFS
    (( ${#SELECTED_JOB_KEYS[@]} > 0 )) || die "No job keys provided to --jobs."
    local key
    for key in "${SELECTED_JOB_KEYS[@]}"; do
        [[ -n "$key" ]] || die "Empty job key in --jobs."
        is_valid_job_key "$key" || die "Unknown job key '$key' in --jobs."
        JOB_FILTER_SET["$key"]=1
    done
}

parse_gpu_ids_arg() {
    local spec="$1"
    [[ -n "$spec" ]] || die "GPU ID list cannot be empty."
    local -a parsed=()
    local -A seen=()
    local id trimmed
    IFS=',' read -r -a parsed <<< "$spec"
    unset IFS
    GPU_ID_LIST=()
    for id in "${parsed[@]}"; do
        trimmed="${id//[[:space:]]/}"
        [[ "$trimmed" =~ ^[0-9]+$ ]] || die "GPU ID '$trimmed' must be a non-negative integer."
        [[ -z "${seen[$trimmed]+x}" ]] || die "Duplicate GPU ID '$trimmed' in --gpu-ids."
        seen["$trimmed"]=1
        GPU_ID_LIST+=("$trimmed")
    done
}

parse_hydra_override_string() {
    local spec="$1"
    [[ -n "$spec" ]] || die "Hydra override string cannot be empty."
    local entry trimmed
    IFS=',' read -r -a entries <<< "$spec"
    unset IFS
    for entry in "${entries[@]}"; do
        trimmed="${entry#"${entry%%[![:space:]]*}"}"
        trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"
        [[ -z "$trimmed" ]] && continue
        USER_HYDRA_OVERRIDES+=("$trimmed")
    done
}

should_run_job() {
    local key="$1"
    (( JOB_FILTER_ENABLED == 0 )) && return 0
    [[ -n "${JOB_FILTER_SET[$key]+x}" ]]
}

get_job_python() {
    local key="$1"
    if [[ -n "${JOB_PYTHON_OVERRIDES[$key]+x}" ]]; then
        echo "${JOB_PYTHON_OVERRIDES[$key]}"
    else
        echo "$DEFAULT_PYTHON"
    fi
}

handle_python_arg() {
    local spec="$1"
    if [[ "$spec" == *"="* ]]; then
        parse_job_python_arg "$spec"
    else
        [[ -n "$spec" ]] || die "Python interpreter path cannot be empty."
        DEFAULT_PYTHON="$spec"
    fi
}

detect_default_python() {
    if [[ -n "${PYTHON_INTERPRETER:-}" ]]; then
        echo "${PYTHON_INTERPRETER}"
    elif command -v python3 >/dev/null 2>&1; then
        command -v python3
    elif command -v python >/dev/null 2>&1; then
        command -v python
    else
        echo "python3"
    fi
}

detect_total_gpus() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d '[:space:]'
    else
        echo "0"
    fi
}

sort_available_gpus() {
    if (( ${#available_gpus[@]} > 1 )); then
        IFS=$'\n' available_gpus=($(printf '%s\n' "${available_gpus[@]}" | sort -n))
        unset IFS
    fi
}

init_gpu_pool() {
    local total="$1"
    local idx
    available_gpus=()
    if (( ${#GPU_ID_LIST[@]} > 0 )); then
        available_gpus=("${GPU_ID_LIST[@]}")
    else
        for ((idx = 0; idx < total; idx++)); do
            available_gpus+=("$idx")
        done
    fi
    sort_available_gpus
}

gpu_is_idle() {
    if (( ALLOW_SHARED_GPU == 1 || NVIDIA_SMI_AVAILABLE == 0 )); then
        return 0
    fi
    local gpu="$1"
    local output trimmed
    if ! output="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$gpu" 2>/dev/null)"; then
        return 0
    fi
    trimmed="${output//[[:space:]]/}"
    [[ -z "$trimmed" || "$trimmed" =~ ^No[Rr]unning[Pp]rocesses[Ff]ound.*$ ]]
}

release_gpus() {
    available_gpus+=("$@")
    sort_available_gpus
}

reap_jobs() {
    local pid exit_code assigned job_name
    for pid in "${!RUNNING_GPU_ASSIGNMENTS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            continue
        fi
        if wait "$pid"; then
            exit_code=0
        else
            exit_code=$?
        fi
        assigned="${RUNNING_GPU_ASSIGNMENTS[$pid]}"
        job_name="${RUNNING_JOB_NAMES[$pid]}"
        unset RUNNING_GPU_ASSIGNMENTS["$pid"]
        unset RUNNING_JOB_NAMES["$pid"]
        if (( exit_code == 0 )); then
            # shellcheck disable=SC2086
            release_gpus $assigned
            log "[$job_name] completed successfully."
        else
            log "[$job_name] failed with exit code $exit_code."
            cleanup_and_exit "$exit_code"
        fi
    done
}

acquire_gpus() {
    local needed="$1"
    local gpu
    while (( ${#available_gpus[@]} < needed )); do
        (( ${#RUNNING_GPU_ASSIGNMENTS[@]} > 0 )) || die "Job requests $needed GPU(s), but only $TOTAL_GPUS available."
        wait -n "${!RUNNING_GPU_ASSIGNMENTS[@]}" 2>/dev/null || true
        reap_jobs
    done
    while true; do
        local selected=()
        local remaining=()
        for gpu in "${available_gpus[@]}"; do
            if (( ${#selected[@]} < needed )) && gpu_is_idle "$gpu"; then
                selected+=("$gpu")
            else
                remaining+=("$gpu")
            fi
        done
        if (( ${#selected[@]} == needed )); then
            ACQUIRED_GPUS=("${selected[@]}")
            available_gpus=("${remaining[@]}")
            sort_available_gpus
            return
        fi
        if (( ${#RUNNING_GPU_ASSIGNMENTS[@]} > 0 )); then
            wait -n "${!RUNNING_GPU_ASSIGNMENTS[@]}" 2>/dev/null || true
            reap_jobs
        else
            sleep "$GPU_IDLE_CHECK_INTERVAL"
        fi
    done
}

cleanup_and_exit() {
    local exit_code="$1"
    (( CLEANUP_IN_PROGRESS )) && exit "$exit_code"
    CLEANUP_IN_PROGRESS=1
    trap - INT TERM
    local pid
    for pid in "${!RUNNING_GPU_ASSIGNMENTS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${!RUNNING_GPU_ASSIGNMENTS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    exit "$exit_code"
}

handle_interrupt() {
    log "Interrupt received. Terminating running jobs..."
    cleanup_and_exit 130
}

run_inference() {
    local prerequisite="$1"
    shift
    if [[ ! -f "$prerequisite" ]]; then
        die "Required question JSONL not found: $prerequisite"
    fi
    "$@"
}

launch_job() {
    local job_key="$1"
    shift
    local requested="${JOB_GPU_COUNTS[$job_key]}"
    [[ "$requested" =~ ^[0-9]+$ ]] || die "GPU count for '$job_key' is not a positive integer."
    (( requested > 0 )) || die "GPU count for '$job_key' must be at least 1."
    (( requested <= TOTAL_GPUS )) || die "Job '$job_key' requests $requested GPU(s), but only $TOTAL_GPUS available."

    reap_jobs
    acquire_gpus "$requested"

    local assigned_csv
    local IFS=,
    assigned_csv="${ACQUIRED_GPUS[*]}"
    unset IFS

    local log_file_basename="$job_key"
    if [[ -n "$RUN_TAG" ]]; then
        log_file_basename="${RUN_TAG}_${log_file_basename}"
    fi
    local log_file="$LOG_DIR/${log_file_basename}_${RUN_TIMESTAMP}.log"

    log "[$job_key] Launching on GPU(s): $assigned_csv"
    log "[$job_key] Logging output to $log_file"
    (
        export CUDA_VISIBLE_DEVICES="$assigned_csv"
        export HF_HOME TRANSFORMERS_CACHE HF_HUB_CACHE
        local cmd=("$@")
        if (( ${#COMMON_HYDRA_OVERRIDES[@]} > 0 )); then
            cmd+=("${COMMON_HYDRA_OVERRIDES[@]}")
        fi
        "${cmd[@]}" >"$log_file" 2>&1
    ) &

    local pid=$!
    RUNNING_GPU_ASSIGNMENTS["$pid"]="${ACQUIRED_GPUS[*]}"
    RUNNING_JOB_NAMES["$pid"]="$job_key"
}

wait_for_all_jobs() {
    while (( ${#RUNNING_GPU_ASSIGNMENTS[@]} > 0 )); do
        wait -n "${!RUNNING_GPU_ASSIGNMENTS[@]}" 2>/dev/null || true
        reap_jobs
    done
}

if (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 3) )); then
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
        print_usage
        exit 0
    fi
    die "This launcher requires Bash 4.3 or newer. Re-run with a newer bash executable."
fi

declare -A JOB_DEFAULT_GPUS=()
declare -A JOB_GPU_OVERRIDES=()
declare -A JOB_MIN_GPU_OVERRIDES=()
declare -A JOB_MIN_GPUS=()
declare -A JOB_GPU_COUNTS=()
declare -A JOB_PYTHON_OVERRIDES=()
declare -A RUNNING_GPU_ASSIGNMENTS=()
declare -A RUNNING_JOB_NAMES=()
declare -A JOB_FILTER_SET=()
declare -a GPU_ID_LIST=()
declare -a available_gpus=()
declare -a ACQUIRED_GPUS=()
declare -a COMMON_HYDRA_OVERRIDES=()
declare -a USER_HYDRA_OVERRIDES=()
declare -a SELECTED_JOB_KEYS=()

TOTAL_GPUS_OVERRIDE=""
TOTAL_GPUS_ENV="${TOTAL_GPUS:-}"
TOTAL_GPUS=""
MODEL="${MODEL:-qwen_2_5_vl}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-VL-72B-Instruct}"
RUN_TAG="${RUN_TAG:-}"
DATA_ROOT="${DATA_ROOT:-data}"
RESULTS_ROOT="${RESULTS_ROOT:-src/eval/results/run_model}"
LOG_DIR="${LOG_DIR:-src/eval/logs/run_model}"
HF_HOME="${HF_HOME:-$(resolve_path "$DATA_ROOT" "hf-cache")}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$(resolve_path "$HF_HOME" "transformers")}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$(resolve_path "$HF_HOME" "hub")}"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_PYTHON="${DEFAULT_PYTHON:-$(detect_default_python)}"
GPU_IDLE_CHECK_INTERVAL="${GPU_IDLE_CHECK_INTERVAL:-10}"
NVIDIA_SMI_AVAILABLE=0
ALLOW_SHARED_GPU=0
JOB_FILTER_ENABLED=0
CLEANUP_IN_PROGRESS=0

trap handle_interrupt INT TERM

while [[ $# -gt 0 ]]; do
    case "$1" in
        --total-gpus) TOTAL_GPUS_OVERRIDE="${2:?Missing value for --total-gpus.}"; shift 2 ;;
        --gpu-ids) parse_gpu_ids_arg "${2:?Missing value for --gpu-ids.}"; shift 2 ;;
        --job-gpus) parse_job_gpus_arg "${2:?Missing value for --job-gpus.}"; shift 2 ;;
        --job-min-gpus) parse_job_min_gpus_arg "${2:?Missing value for --job-min-gpus.}"; shift 2 ;;
        --python) handle_python_arg "${2:?Missing value for --python.}"; shift 2 ;;
        --allow-shared-gpu) ALLOW_SHARED_GPU=1; shift ;;
        --jobs) parse_selected_jobs_arg "${2:?Missing value for --jobs.}"; shift 2 ;;
        --model) MODEL="${2:?Missing value for --model.}"; shift 2 ;;
        --model-name) MODEL_NAME="${2:?Missing value for --model-name.}"; shift 2 ;;
        --hydra-override) parse_hydra_override_string "${2:?Missing value for --hydra-override.}"; shift 2 ;;
        --hydra-overrides) parse_hydra_override_string "${2:?Missing value for --hydra-overrides.}"; shift 2 ;;
        --run-tag) RUN_TAG="${2:?Missing value for --run-tag.}"; shift 2 ;;
        --log-dir) LOG_DIR="${2:?Missing value for --log-dir.}"; shift 2 ;;
        --data-root) DATA_ROOT="${2:?Missing value for --data-root.}"; shift 2 ;;
        --results-root) RESULTS_ROOT="${2:?Missing value for --results-root.}"; shift 2 ;;
        -h|--help) print_usage; exit 0 ;;
        *) die "Unknown argument: $1" ;;
    esac
done

if (( ${#GPU_ID_LIST[@]} > 0 )); then
    [[ -z "$TOTAL_GPUS_OVERRIDE" ]] || die "Cannot combine --gpu-ids with --total-gpus."
    TOTAL_GPUS="${#GPU_ID_LIST[@]}"
elif [[ -n "$TOTAL_GPUS_OVERRIDE" ]]; then
    TOTAL_GPUS="$TOTAL_GPUS_OVERRIDE"
elif [[ -n "$TOTAL_GPUS_ENV" ]]; then
    TOTAL_GPUS="$TOTAL_GPUS_ENV"
else
    TOTAL_GPUS="$(detect_total_gpus)"
fi
[[ "$TOTAL_GPUS" =~ ^[0-9]+$ ]] || die "Total GPU count '$TOTAL_GPUS' is not a non-negative integer."
(( TOTAL_GPUS > 0 )) || die "Total GPU count must be at least 1. Pass --total-gpus if auto-detection is unavailable."

if command -v nvidia-smi >/dev/null 2>&1; then
    NVIDIA_SMI_AVAILABLE=1
else
    log "nvidia-smi not found; skipping external GPU occupancy checks."
fi

num_jobs=${#JOB_KEYS[@]}
base_gpus=$(( TOTAL_GPUS / num_jobs ))
remainder=$(( TOTAL_GPUS % num_jobs ))
(( base_gpus > 0 )) || base_gpus=1
for idx in "${!JOB_KEYS[@]}"; do
    key="${JOB_KEYS[$idx]}"
    JOB_DEFAULT_GPUS["$key"]="$base_gpus"
    if (( remainder > 0 )); then
        JOB_DEFAULT_GPUS["$key"]=$(( ${JOB_DEFAULT_GPUS[$key]} + 1 ))
        remainder=$(( remainder - 1 ))
    fi
done

for key in "${JOB_KEYS[@]}"; do
    JOB_MIN_GPUS["$key"]="${JOB_MIN_GPU_OVERRIDES[$key]:-1}"
    (( JOB_MIN_GPUS["$key"] > 0 )) || die "Minimum GPU count for '$key' must be at least 1."
    (( JOB_MIN_GPUS["$key"] <= TOTAL_GPUS )) || die "Minimum GPU requirement for '$key' exceeds available GPUs."
    if [[ -n "${JOB_GPU_OVERRIDES[$key]+x}" ]]; then
        JOB_GPU_COUNTS["$key"]="${JOB_GPU_OVERRIDES[$key]}"
    elif (( ${JOB_DEFAULT_GPUS[$key]} < ${JOB_MIN_GPUS[$key]} )); then
        JOB_GPU_COUNTS["$key"]="${JOB_MIN_GPUS[$key]}"
    else
        JOB_GPU_COUNTS["$key"]="${JOB_DEFAULT_GPUS[$key]}"
    fi
done

build_common_hydra_overrides
mkdir -p "$LOG_DIR" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE"
init_gpu_pool "$TOTAL_GPUS"

MEDIA_CACHE_DIR_ROOT="$(resolve_path "$RESULTS_ROOT" "media_cache")"
RENDERED_TEMPLATE_DIR_ROOT="$(resolve_path "$RESULTS_ROOT" "rendered_templates")"
RESPONSES_DIR_ROOT="$(resolve_path "$RESULTS_ROOT" "responses")"

keyframe_video_dir="$(resolve_path "$DATA_ROOT" "videos/keyframe-video/fps-1")"
trimmed_video_dir="$(resolve_path "$DATA_ROOT" "videos/trimmed-videos")"
mask_dir="$(resolve_path "$DATA_ROOT" "segmentation-masks")"
img_dir="$(resolve_path "$DATA_ROOT" "rgb-frames")"
question_dir="$(resolve_path "$DATA_ROOT" "questions/yamls")"
question_jsonl_fn="$(resolve_path "$DATA_ROOT" "questions/questions.jsonl")"

declare -A JOB_CONFIG_NAMES=(
    ["sep_media_first_keyframe"]="qwen_2_5_vl_sep_media_first"
    ["concat_media_first_keyframe"]="qwen_2_5_vl_concat_media_first"
    ["collage_media_first_keyframe"]="qwen_2_5_vl_collage_media_first"
    ["sep_media_first_trimmed"]="qwen_2_5_vl_sep_media_first"
    ["concat_media_first_trimmed"]="qwen_2_5_vl_concat_media_first"
    ["collage_media_first_trimmed"]="qwen_2_5_vl_collage_media_first"
)
declare -A JOB_VIDEO_DIRS=(
    ["sep_media_first_keyframe"]="$keyframe_video_dir"
    ["concat_media_first_keyframe"]="$keyframe_video_dir"
    ["collage_media_first_keyframe"]="$keyframe_video_dir"
    ["sep_media_first_trimmed"]="$trimmed_video_dir"
    ["concat_media_first_trimmed"]="$trimmed_video_dir"
    ["collage_media_first_trimmed"]="$trimmed_video_dir"
)

log "Logs will be written to: $LOG_DIR"
log "Detected $TOTAL_GPUS GPU(s)."
log "MODEL: $MODEL"
log "MODEL_NAME: $MODEL_NAME"

for key in "${JOB_KEYS[@]}"; do
    if ! should_run_job "$key"; then
        log "[$key] Skipped by --jobs filter."
        continue
    fi
    python_bin="$(get_job_python "$key")"
    response_name="$(format_response_dir_name "${key}_video")"
    launch_job "$key" \
        run_inference "$question_jsonl_fn" \
        "$python_bin" src/eval/inference.py --config-name "${JOB_CONFIG_NAMES[$key]}" \
            "video_dir=${JOB_VIDEO_DIRS[$key]}" \
            "img_dir=$img_dir" \
            "mask_dir=$mask_dir" \
            "question_dir=$question_dir" \
            "question_jsonl_fn=$question_jsonl_fn" \
            "responses_dir=$(resolve_path "$RESPONSES_DIR_ROOT" "$response_name")" \
            "media_cache_dir=$(resolve_path "$MEDIA_CACHE_DIR_ROOT" "${key}_video")" \
            "rendered_template_dir=$(resolve_path "$RENDERED_TEMPLATE_DIR_ROOT" "${key}_video")"
done

wait_for_all_jobs
log "All inference jobs completed."
