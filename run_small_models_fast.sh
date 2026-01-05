#!/bin/bash
#SBATCH --job-name=visa_small_fast
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --output=logs/visa_small_%j.out
#SBATCH --error=logs/visa_small_%j.err

# ============================================================================
# OPTIMIZED SCRIPT: Small models (4B-14B) with proper 1-GPU allocation
# ============================================================================
# 
# This runs small models SEQUENTIALLY but efficiently on 1 GPU each
# Much faster than using tensor_parallel=4 unnecessarily
# 
# Estimated time: ~6-8 hours for 50 runs (vs 15+ hours with old config)
# ============================================================================

echo "=========================================="
echo "Running Small Models (Optimized - 1 GPU each)"
echo "=========================================="
echo "Models: qwen3-4b qwen3-8b ministral-8b llama31-8b ministral-14b-reasoning gemma2-9b"
echo "Runs per candidate: 50"
echo "Each model uses 1 GPU (except Gemma uses eager mode)"
echo "=========================================="
echo ""

# Export the models list
export MODELS="qwen3-4b qwen3-8b ministral-8b llama31-8b ministral-14b-reasoning gemma2-9b"
export RUNS=50

# Run using the fixed custom mode
bash run_experiment.sh custom

