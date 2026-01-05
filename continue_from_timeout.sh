#!/bin/bash
#SBATCH --job-name=visa_continue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:2
#SBATCH --output=logs/visa_continue_%j.out
#SBATCH --error=logs/visa_continue_%j.err

# ============================================================================
# SMART RESUME: Continue from Job 2701964 timeout (or any partial results)
# ============================================================================
# 
# This script:
# 1. Finds the most recent results file
# 2. Analyzes what's been completed
# 3. Automatically resumes with remaining models
# 4. Uses optimized GPU allocation per model
# 
# Estimated time: 8-12 hours (depending on what was already done)
# ============================================================================

echo "=========================================="
echo "SMART RESUME: Continuing from timeout"
echo "=========================================="
echo ""

# Find workspace directory
WORKSPACE_DIR="/pfs/work9/workspace/scratch/ma_anhnnguy-topic_modeling_data/RAI"
cd "$WORKSPACE_DIR" || {
    echo "ERROR: Workspace not found: $WORKSPACE_DIR"
    exit 1
}

# Find the most recent results file
LATEST_RESULTS=$(ls -t results/results_20*.json 2>/dev/null | head -1)

if [ -z "$LATEST_RESULTS" ]; then
    echo "❌ No previous results found in results/"
    echo ""
    echo "Options:"
    echo "  1. Run fresh start: sbatch run_small_models_fast.sh"
    echo "  2. Manually specify: --resume-from <file>"
    exit 1
fi

echo "📂 Found previous results:"
echo "   $LATEST_RESULTS"
echo ""

# Analyze what's completed
echo "🔍 Analyzing completed models..."
python check_completed_models.py "$LATEST_RESULTS"
echo ""

# Define all models we want
ALL_MODELS="gemma2-9b gemma2-27b llama31-8b llama31-70b ministral-8b ministral-14b-reasoning mistral-small qwen3-4b qwen3-8b qwen3-30b qwen3-32b"

echo "=========================================="
echo "Starting resume with optimized settings"
echo "=========================================="
echo "Resume from: $LATEST_RESULTS"
echo "All models: $ALL_MODELS"
echo "Runs: ${RUNS:-300}"
echo ""
echo "The script will automatically:"
echo "  ✓ Load existing results"
echo "  ✓ Skip completed models"
echo "  ✓ Run each remaining model with optimal GPU config"
echo "  ✓ Save incrementally (safe from timeout)"
echo "=========================================="
echo ""

# Export configuration
export MODELS="$ALL_MODELS"
export RUNS=${RUNS:-300}
export RESUME_FILE="$LATEST_RESULTS"

# Run with custom mode (now has per-model optimization)
echo "Executing optimized run..."
echo ""

python run_bias_evaluation.py \
    --resume-from "$LATEST_RESULTS" \
    --config config.yaml \
    --models $ALL_MODELS \
    --runs ${RUNS:-300} \
    --output results \
    --gpu-memory 0.85 \
    --tensor-parallel 2

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Resume completed successfully!"
    echo ""
    echo "Final results saved in: results/"
    ls -lht results/*.json | head -5
else
    echo "✗ Resume failed with exit code: $EXIT_CODE"
    echo ""
    echo "Check logs:"
    echo "  logs/visa_continue_${SLURM_JOB_ID}.err"
fi
echo "=========================================="

exit $EXIT_CODE

