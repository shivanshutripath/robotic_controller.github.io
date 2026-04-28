#!/bin/bash
# compare_models.sh

CSV_LOG="metrics_model_comparison.csv"
USER_PROMPT="Starting at (-0.53,-0.52), and Goal at (0.5,-0.05). Generate controllers/obs_avoidance/obs_avoidance.py using the project contracts and the provided world file."

# # GPT-4o (5 runs)
# echo "Running GPT-4o experiments..."
# for i in {1..5}; do
#   python tools/repair.py \
#     --code-gen tools/code_gen.py \
#     --prompt tools/prompt.py \
#     --world worlds/empty.wbt \
#     --out controllers/obs_avoidance/obs_avoidance.py \
#     --tests tests \
#     --model gpt-4o \
#     --max-iters 20 \
#     --csv-log $CSV_LOG \
#     --run-id codegen_4o_$i \
#     --user-prompt "$USER_PROMPT"
# done

# # GPT-4.1 (5 runs)
# echo "Running GPT-4.1 experiments..."
# for i in {1..5}; do
#   python tools/repair.py \
#     --code-gen tools/code_gen.py \
#     --prompt tools/prompt.py \
#     --world worlds/empty.wbt \
#     --out controllers/obs_avoidance/obs_avoidance.py \
#     --tests tests \
#     --model gpt-4.1 \
#     --max-iters 20 \
#     --csv-log $CSV_LOG \
#     --run-id codegen_4p1_$i \
#     --user-prompt "$USER_PROMPT"
# done

# # GPT-5.2 (5 runs)
# echo "Running GPT-5.2 experiments..."
# for i in {1..5}; do
#   python tools/repair.py \
#     --code-gen tools/code_gen.py \
#     --prompt tools/prompt.py \
#     --world worlds/empty.wbt \
#     --out controllers/obs_avoidance/obs_avoidance.py \
#     --tests tests \
#     --model gpt-5.2 \
#     --max-iters 20 \
#     --csv-log $CSV_LOG \
#     --run-id codegen_5p2_$i \
#     --user-prompt "$USER_PROMPT"
# done

# # Claude Opus 4.5 (5 runs)
# echo "Running Claude Opus 4.5 experiments..."
# for i in {1..5}; do
#   python tools/repair.py \
#     --code-gen tools/code_gen.py \
#     --prompt tools/prompt.py \
#     --world worlds/empty.wbt \
#     --out controllers/obs_avoidance/obs_avoidance.py \
#     --tests tests \
#     --model claude-opus-4-5-20251101 \
#     --max-iters 20 \
#     --csv-log $CSV_LOG \
#     --run-id codegen_claude_opus_4p5_$i \
#     --user-prompt "$USER_PROMPT"
# done

# # Claude Sonnet 4.5 (5 runs)
# echo "Running Claude Sonnet 4.5 experiments..."
# for i in {1..5}; do
#   python tools/repair.py \
#     --code-gen tools/code_gen.py \
#     --prompt tools/prompt.py \
#     --world worlds/empty.wbt \
#     --out controllers/obs_avoidance/obs_avoidance.py \
#     --tests tests \
#     --model claude-sonnet-4-5-20250929 \
#     --max-iters 20 \
#     --csv-log $CSV_LOG \
#     --run-id codegen_claude_sonnet_4p5_$i \
#     --user-prompt "$USER_PROMPT"
# done

# # Claude Haiku 4.5 (5 runs)
# echo "Running Claude Haiku 4.5 experiments..."
# for i in {1..5}; do
#   python tools/repair.py \
#     --code-gen tools/code_gen.py \
#     --prompt tools/prompt.py \
#     --world worlds/empty.wbt \
#     --out controllers/obs_avoidance/obs_avoidance.py \
#     --tests tests \
#     --model claude-haiku-4-5-20251001 \
#     --max-iters 20 \
#     --csv-log $CSV_LOG \
#     --run-id codegen_claude_haiku_4p5_$i \
#     --user-prompt "$USER_PROMPT"
# done

# Claude Opus 4.5 (5 runs)
echo "Running Claude Opus 4.5 experiments..."
for i in {1..5}; do
  python tools/repair.py \
    --code-gen tools/code_gen.py \
    --prompt tools/prompt.py \
    --world worlds/empty.wbt \
    --out controllers/obs_avoidance/obs_avoidance.py \
    --tests tests \
    --model claude-opus-4-5-20251101 \
    --max-iters 20 \
    --csv-log $CSV_LOG \
    --run-id codegen_claude_opus_4p5_$i \
    --user-prompt "$USER_PROMPT"
done

echo "All experiments completed!"
echo "Results saved to: $CSV_LOG"
echo ""
echo "Generate comparison plots:"
echo "  python plot_metrics.py --csv $CSV_LOG --save-dir plots"