#!/bin/bash
#SBATCH --job-name=visa_large_fast
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:4
#SBATCH --output=logs/visa_large_%j.out
#SBATCH --error=logs/visa_large_%j.err

# ============================================================================
# OPTIMIZED SCRIPT: Large models (27B-32B) with proper 2-GPU allocation
# ============================================================================
# 
# This runs large models with tensor_parallel=2 (optimal for A100 40GB)
# Gemma models automatically get --enforce-eager flag
# 
# Estimated time: ~8-10 hours for 50 runs
# ============================================================================

echo "=========================================="
echo "Running Large Models (Optimized - 2 GPUs each)"
echo "=========================================="
echo "Models: gemma2-27b qwen3-30b qwen3-32b mistral-small"
echo "Runs per candidate: 50"
echo "Each model uses 2 GPUs (tensor_parallel=2)"
echo "Gemma models automatically use eager mode"
echo "=========================================="
echo ""

# Export the models list
export MODELS="gemma2-27b qwen3-30b qwen3-32b mistral-small"
export RUNS=50

# Run using the fixed custom mode
bash run_experiment.sh custom

x