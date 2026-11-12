#!/usr/bin/env bash
# Sequential remote training/evaluation with experiment-level restart recovery.
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd -P)
CONFIG_DIR="$PROJECT_ROOT/configs"
OUTPUT_ROOT="/root/autodl-fs/model_v2_artifact/ablations"
UV_BIN=${UV_EXECUTABLE:-/root/.local/bin/uv}
PYTHON_BIN=${PYTHON_EXECUTABLE:-$PROJECT_ROOT/.venv/bin/python}
STATE_TOOL="$SCRIPT_DIR/ablation_state.py"
LANGUAGE=csharp
TEST_DATA=""
REVISION=""
TRAIN_ONLY=0
DRY_RUN=0
INCLUDE_MAIN=0
FAIL_FAST=0
SELECTED=()
FORCE_NAMES=()

usage() {
    cat <<'HELP'
Usage: bash scripts/run_ablations.sh [options] [config names or paths...]

Default order: frozen.json, fullonly.json, diffonly.json, mean.json, concat.json
  --include-main        Run qwen3_06b.json before the default ablations.
  --config-dir DIR      Directory containing the experiment configs.
  --output-root DIR     Stable persistent batch directory (reuse to recover).
  --train-only          Train/calibrate only; defer test evaluation.
  --test-data PATH      Override the test file; it must be declared in manifest.
  --language LANGUAGE   csharp (default), java or cpp.
  --revision SHA        Pin both model/tokenizer revisions for every experiment.
  --rerun NAME          Intentionally retrain this experiment, keeping old outputs.
  --fail-fast           Stop at the first failure (default: continue the queue).
  --dry-run             Show plans without running models or writing batch state.
  -h, --help            Show this help.

Environment: UV_EXECUTABLE (default /root/.local/bin/uv),
PYTHON_EXECUTABLE (default project .venv/bin/python),
HF_ENDPOINT (defaults to https://hf-mirror.com).

Recovery is per experiment/stage, not optimizer/epoch continuation. A failed
training starts a new attempt; failed evaluation reuses complete trained weights.
Completed unchanged experiments are skipped. Partial outputs are never deleted.
HELP
}

need_value() {
    if (($# < 2)) || [[ -z $2 ]]; then
        printf 'Missing value for %s\n' "$1" >&2
        exit 2
    fi
}

while (($#)); do
    case "$1" in
        --include-main) INCLUDE_MAIN=1; shift ;;
        --config-dir) need_value "$@"; CONFIG_DIR=$2; shift 2 ;;
        --output-root) need_value "$@"; OUTPUT_ROOT=$2; shift 2 ;;
        --test-data) need_value "$@"; TEST_DATA=$2; shift 2 ;;
        --language) need_value "$@"; LANGUAGE=$2; shift 2 ;;
        --revision) need_value "$@"; REVISION=$2; shift 2 ;;
        --rerun) need_value "$@"; FORCE_NAMES+=("${2%.json}"); shift 2 ;;
        --train-only) TRAIN_ONLY=1; shift ;;
        --fail-fast) FAIL_FAST=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; SELECTED+=("$@"); break ;;
        -*) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
        *) SELECTED+=("$1"); shift ;;
    esac
done

cd -- "$PROJECT_ROOT"
[[ -x $PYTHON_BIN ]] || { printf 'Python missing: %s; install the project environment first.\n' "$PYTHON_BIN" >&2; exit 2; }
if ((!DRY_RUN)); then
    [[ -x $UV_BIN ]] || { printf 'uv missing: %s\n' "$UV_BIN" >&2; exit 2; }
    for utility in flock setsid tee; do
        command -v "$utility" >/dev/null || { printf 'Required utility missing: %s\n' "$utility" >&2; exit 2; }
    done
    mkdir -p -- "$OUTPUT_ROOT"
    OUTPUT_ROOT=$(cd -- "$OUTPUT_ROOT" && pwd -P)
    exec 9>"$OUTPUT_ROOT/.batch.lock"
    flock -n 9 || { printf 'Another batch already owns %s\n' "$OUTPUT_ROOT" >&2; exit 2; }
fi
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTHONUNBUFFERED=1

if ((${#SELECTED[@]} == 0)); then
    SELECTED=(frozen fullonly diffonly mean concat)
fi
if ((INCLUDE_MAIN)); then
    SELECTED=(qwen3_06b "${SELECTED[@]}")
fi
declare -A SEEN=()
CONFIGS=()
for selected in "${SELECTED[@]}"; do
    if [[ $selected == */* ]]; then
        config=$selected
    else
        config="$CONFIG_DIR/${selected%.json}.json"
    fi
    name=${config##*/}
    name=${name%.json}
    if [[ -n ${SEEN[$name]+yes} ]]; then
        printf 'Duplicate experiment name: %s\n' "$name" >&2
        exit 2
    fi
    SEEN[$name]=1
    CONFIGS+=("$config")
done
for name in "${FORCE_NAMES[@]}"; do
    [[ -n ${SEEN[$name]+yes} ]] || { printf -- '--rerun name is not in this queue: %s\n' "$name" >&2; exit 2; }
done

PLAN_FILE=$(mktemp)
ACTIVE_PID=""
ACTIVE_JOB=""
ACTIVE_PHASE=""
cleanup() { rm -f -- "$PLAN_FILE"; }
trap cleanup EXIT

log() {
    local message
    printf -v message '[%s] %s' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
    if ((DRY_RUN)); then printf '%s\n' "$message";
    else printf '%s\n' "$message" | tee -a "$OUTPUT_ROOT/batch.log"; fi
}

interrupted() {
    local code=$1
    trap '' INT TERM
    if [[ -n $ACTIVE_PID ]]; then
        kill -TERM -- "-$ACTIVE_PID" 2>/dev/null || true
        wait "$ACTIVE_PID" 2>/dev/null || true
        "$PYTHON_BIN" "$STATE_TOOL" mark --job "$ACTIVE_JOB" --phase "$ACTIVE_PHASE" --exit-code "$code" || true
    fi
    log "Interrupted. Restart with the same output root to recover: $OUTPUT_ROOT"
    exit "$code"
}
trap 'interrupted 130' INT
trap 'interrupted 143' TERM

run_stage() {
    local phase=$1 job=$2 console=$3 code
    shift 3
    "$PYTHON_BIN" "$STATE_TOOL" mark --job "$job" --phase "$phase" --running --pid "$$" || return 1
    ACTIVE_JOB=$job
    ACTIVE_PHASE=$phase
    setsid "$UV_BIN" run --no-sync model "$@" >"$console" 2>&1 &
    ACTIVE_PID=$!
    if wait "$ACTIVE_PID"; then code=0; else code=$?; fi
    ACTIVE_PID=""
    if "$PYTHON_BIN" "$STATE_TOOL" mark --job "$job" --phase "$phase" --exit-code "$code"; then
        return 0
    fi
    log "FAILED $phase (exit=$code). Console log: $console"
    if [[ -f $console ]] && "$PYTHON_BIN" - "$console" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
with path.open('rb') as stream:
    stream.seek(max(0, path.stat().st_size - 65536))
    text = stream.read().decode('utf-8', errors='replace').lower()
sys.exit(0 if any(word in text for word in ('cuda out of memory', 'cuda.outofmemoryerror', 'resource failure')) else 1)
PY
    then
        log 'Resource failure: increase GPU/memory capacity before retrying; no input windows were silently dropped.'
    fi
    return 1
}

completed=0
skipped=0
failed=0
log "Batch start. Output root: $OUTPUT_ROOT"
for config in "${CONFIGS[@]}"; do
    name=${config##*/}
    name=${name%.json}
    force=0
    for requested in "${FORCE_NAMES[@]}"; do
        if [[ $name == "$requested" ]]; then force=1; fi
    done
    attempted_train=0
    attempted_eval=0
    changed=0
    ok=1
    while true; do
        plan=("$PYTHON_BIN" "$STATE_TOOL" plan --config "$config" --root "$OUTPUT_ROOT" --language "$LANGUAGE")
        [[ -z $TEST_DATA ]] || plan+=(--test-data "$TEST_DATA")
        [[ -z $REVISION ]] || plan+=(--revision "$REVISION")
        ((TRAIN_ONLY == 0)) || plan+=(--train-only)
        ((DRY_RUN == 0)) || plan+=(--dry-run)
        ((force == 0)) || plan+=(--force)
        force=0
        if ! "${plan[@]}" >"$PLAN_FILE"; then
            log "FAILED preparing $name. Other completed experiments remain available."
            ok=0
            break
        fi
        mapfile -d '' -t fields <"$PLAN_FILE"
        case ${fields[0]} in
            skip)
                if ((changed)); then log "DONE $name"; completed=$((completed+1));
                else log "SKIP $name: complete artifacts match config/data."; skipped=$((skipped+1)); fi
                break ;;
            train)
                if ((attempted_train)); then log "FAILED $name: inputs/artifacts changed during this invocation; restart after fixing."; ok=0; break; fi
                attempted_train=1
                log "TRAIN $name. Console log: ${fields[4]}"
                command=(train --config "${fields[2]}" --output "${fields[3]}")
                if ((DRY_RUN)); then printf '  %q ' "$UV_BIN" run --no-sync model "${command[@]}"; printf '\n'; break; fi
                if ! run_stage train "${fields[1]}" "${fields[4]}" "${command[@]}"; then ok=0; break; fi
                changed=1 ;;
            eval)
                if ((attempted_eval)); then log "FAILED $name: evaluation artifacts did not remain complete; restart after fixing."; ok=0; break; fi
                attempted_eval=1
                log "EVAL $name. Console log: ${fields[7]}"
                command=(eval --checkpoint "${fields[2]}" --data "${fields[3]}" --manifest "${fields[4]}"
                         --language "${fields[5]}" --output "${fields[6]}")
                if ((DRY_RUN)); then printf '  %q ' "$UV_BIN" run --no-sync model "${command[@]}"; printf '\n'; break; fi
                if ! run_stage eval "${fields[1]}" "${fields[7]}" "${command[@]}"; then ok=0; break; fi
                changed=1 ;;
            *) log "FAILED $name: invalid state-tool response."; ok=0; break ;;
        esac
    done
    if ((!ok)); then
        failed=$((failed+1))
        if ((FAIL_FAST)); then break; fi
    fi
done
log "Batch finished: completed=$completed skipped=$skipped failed=$failed."
((failed == 0))
