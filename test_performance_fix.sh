#!/bin/bash
#SBATCH --job-name=visa_test_fix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --output=logs/visa_test_%j.out
#SBATCH --error=logs/visa_test_%j.err

# ============================================================================
# QUICK TEST: Verify performance fixes work correctly
# ============================================================================
# 
# This runs 2 small models with just 5 runs to verify:
# 1. Correct GPU allocation (tensor_parallel=1)
# 2. No unnecessary --enforce-eager
# 3. Fast execution time
# 
# Expected time: ~15-20 minutes
# ============================================================================

echo "=========================================="
echo "TESTING Performance Fixes"
echo "=========================================="
echo "Running 2 small models with 5 runs each"
echo "Expected: ~15-20 minutes total"
echo "Watch for correct GPU allocation in logs"
echo "=========================================="
echo ""

# Test with small models only
export MODELS="ministral-8b qwen3-4b"
export RUNS=5

# Run using the fixed custom mode
bash run_experiment.sh custom

echo ""
echo "=========================================="
echo "TEST COMPLETE"
echo "=========================================="
echo ""
echo "Verify in logs above:"
echo "  ✓ Each model shows 'Config: Small model - 1 GPU'"
echo "  ✓ NO 'Extra-large model detected' messages"
echo "  ✓ NO '--enforce-eager' for non-Gemma models"
echo "  ✓ Completed in ~15-20 minutes"
echo ""

