#!/usr/bin/env bash
# ============================================================================
# run.sh — Model Comparison Benchmark
#
# Runs R=5 repetitions (k=0..4) × K=20 max iterations for each model.
# Results are stored in benchmark_results/ with per-model/per-rep artifacts.
#
# Usage:
#   chmod +x run.sh
#   ./run.sh                    # run all models with defaults
#   ./run.sh --models claude-opus-4.5   # run only one model
#   ./run.sh --R 3 --K 10       # fewer reps / iters for a quick test
#
# Prerequisites:
#   - OPENAI_API_KEY must be set (for GPT models)
#   - ANTHROPIC_API_KEY must be set (for Claude models)
#   - Python packages: openai, anthropic, pytest, pytest-json-report, pygame, numpy
#   - Project files: loop_agent.py, code_agent.py, controller_template.py,
#     robot.py, DDR.png, map_agent_outputs/occupancy.png, params.json, tests
# ============================================================================
set -euo pipefail

# ── Defaults ──
# Available models:
# OpenAI: gpt-5.2, gpt-4.1, gpt-4o
# Claude: claude-opus-4.5, claude-sonnet-4.5, claude-haiku-4.5
MODELS="claude-sonnet-4.5,claude-haiku-4.5"
R=5                                       # repetitions per model (k=0..R-1)
K=20                                      # max iterations per run
EDIT_RETRIES=2                            # edit retries per iteration
OPTIMIZER_MODEL="gpt-4o"                  # model for AUTO_REPAIR_RULES optimization
PROJECT_DIR="."                           # project root
OUTPUT_DIR="benchmark_results"            # output directory
TIMEOUT=""                                # per-run timeout (empty = no timeout)
START_REP=0                               # starting repetition index

# ── Parse CLI overrides ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)       MODELS="$2";          shift 2 ;;
        --R)            R="$2";               shift 2 ;;
        --K)            K="$2";               shift 2 ;;
        --edit-retries) EDIT_RETRIES="$2";    shift 2 ;;
        --optimizer)    OPTIMIZER_MODEL="$2"; shift 2 ;;
        --project)      PROJECT_DIR="$2";     shift 2 ;;
        --output)       OUTPUT_DIR="$2";      shift 2 ;;
        --timeout)      TIMEOUT="$2";         shift 2 ;;
        --start-rep)    START_REP="$2";       shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--models M1,M2] [--R 5] [--K 20] [--edit-retries 3]"
            echo "          [--optimizer MODEL] [--project DIR] [--output DIR]"
            echo "          [--timeout SECS] [--start-rep N]"
            echo ""
            echo "Available models:"
            echo "  OpenAI:  gpt-5.2, gpt-4.1, gpt-4o"
            echo "  Claude:  claude-opus-4.5, claude-sonnet-4.5, claude-haiku-4.5"
            echo ""
            echo "Examples:"
            echo "  $0 --models claude-opus-4.5,gpt-4o"
            echo "  $0 --models claude-sonnet-4.5 --R 3"
            exit 0 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Preflight checks ──
# Check if any Claude models are in the list
if echo "$MODELS" | grep -q "claude"; then
    if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        echo "ERROR: ANTHROPIC_API_KEY not set. Claude models require this key."
        echo "Export it with: export ANTHROPIC_API_KEY='your-key-here'"
        exit 1
    fi
fi

# Check if any OpenAI models are in the list
if echo "$MODELS" | grep -qE "gpt-|o1-"; then
    if [ -z "${OPENAI_API_KEY:-}" ]; then
        echo "ERROR: OPENAI_API_KEY not set. OpenAI models require this key."
        echo "Export it with: export OPENAI_API_KEY='your-key-here'"
        exit 1
    fi
fi

if ! python -c "import openai" 2>/dev/null; then
    echo "Installing required packages..."
    pip install openai pytest pytest-json-report pygame numpy --quiet
fi

# Install anthropic if Claude models are being used
if echo "$MODELS" | grep -q "claude"; then
    if ! python -c "import anthropic" 2>/dev/null; then
        echo "Installing anthropic package..."
        pip install anthropic --quiet
    fi
fi

# Ensure benchmark.py is present
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_PY="${SCRIPT_DIR}/benchmark.py"
if [ ! -f "$BENCHMARK_PY" ]; then
    echo "ERROR: benchmark.py not found at $BENCHMARK_PY"
    exit 1
fi

# ── Timestamp for this benchmark session ──
TIMESTAMP=$(date -u +"%Y%m%dT%H%M%SZ")
FULL_OUTPUT="${OUTPUT_DIR}/${TIMESTAMP}"

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  MODEL COMPARISON BENCHMARK"
echo "║  Models:      ${MODELS}"
echo "║  Reps (R):    ${R}  (k=${START_REP}..$(( START_REP + R - 1 )))"
echo "║  Max iters:   ${K}"
echo "║  Edit retries:${EDIT_RETRIES}"
echo "║  Optimizer:   ${OPTIMIZER_MODEL}"
echo "║  Output:      ${FULL_OUTPUT}"
echo "╚══════════════════════════════════════════════════════════════════════╝"

# ── Build command ──
CMD=(
    python "$BENCHMARK_PY"
    --project "$PROJECT_DIR"
    --models "$MODELS"
    --R "$R"
    --K "$K"
    --edit-retries "$EDIT_RETRIES"
    --optimizer-model "$OPTIMIZER_MODEL"
    --output-dir "$FULL_OUTPUT"
    --start-rep "$START_REP"
)

if [ -n "$TIMEOUT" ]; then
    CMD+=(--timeout "$TIMEOUT")
fi

# ── Run ──
echo ""
echo "Running: ${CMD[*]}"
echo ""

"${CMD[@]}" 2>&1 | tee "${FULL_OUTPUT}/benchmark.log"

EXIT_CODE=${PIPESTATUS[0]}

# ── Post-run: create a "latest" symlink ──
LATEST_LINK="${OUTPUT_DIR}/latest"
rm -f "$LATEST_LINK"
ln -sf "$TIMESTAMP" "$LATEST_LINK"

echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  Done! Results in: ${FULL_OUTPUT}"
echo "  Symlink:          ${LATEST_LINK} -> ${TIMESTAMP}"
echo "  Summary CSV:      ${FULL_OUTPUT}/summary.csv"
echo "  Full report:      ${FULL_OUTPUT}/benchmark_report.json"
echo "════════════════════════════════════════════════════════════════════════"

exit $EXIT_CODE