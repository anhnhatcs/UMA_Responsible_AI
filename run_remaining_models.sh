#!/bin/bash
#SBATCH --job-name=visa_wall_remaining
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:2
#SBATCH --output=logs/visa_wall_remaining_%j.out
#SBATCH --error=logs/visa_wall_remaining_%j.err

# ============================================================================
# OPTIMIZED SCRIPT: Run remaining 5 models after Job 2701964 timeout
# ============================================================================
# 
# What this does:
# - Runs mistral-small (24B) - needs 2 GPUs
# - Runs all 4 Qwen models (4B, 8B, 30B, 32B) with correct GPU allocation
# 
# The 30B/32B models will use 2 GPUs, the 4B/8B will use 1 GPU each
# Estimated time: 8-10 hours for 300 runs per candidate
# ============================================================================

echo "=========================================="
echo "Running Remaining Models (Optimized)"
echo "=========================================="
echo "Models: mistral-small qwen3-4b qwen3-8b qwen3-30b qwen3-32b"
echo "Runs per candidate: 300"
echo "Using custom mode with per-model GPU optimization"
echo "=========================================="
echo ""

# Export the models list
export MODELS="mistral-small qwen3-4b qwen3-8b qwen3-30b qwen3-32b"
export RUNS=300

# Run using the fixed custom mode
bash run_experiment.sh custom

