#!/bin/bash
#SBATCH --job-name=visa_wall_bias
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:2
#SBATCH --output=logs/visa_wall_%j.out
#SBATCH --error=logs/visa_wall_%j.err

# ============================================
# SLURM Batch Job Script for Visa Wall Bias Evaluation
# A100 GPU - LLM Bias Testing
# ============================================
#
# Project: The Visa Wall - Benchmarking LLM Bias Against Non-EU Applicants
#
# Models Available:
# - gemma2-9b (9B) - 1 GPU
# - gemma2-27b (27B) - 1 GPU
# - mistral-7b (7B) - 1 GPU
# - mistral-nemo (12B) - 1 GPU
# - qwen3-4b (4B) - 1 GPU
# - qwen3-30b (30B) - 2 GPUs
#
# Usage:
#   sbatch run_experiment.sh test      # Quick test (1 model, 1 run)
#   sbatch run_experiment.sh small     # Small models (8B-14B)
#   sbatch run_experiment.sh qwen      # QWEN models (7B + 72B, needs 2 GPUs for 72B)
#   sbatch --gres=gpu:2 run_experiment.sh large  # Large models (70B+)
#
# ============================================

# ============================================
# Workspace Configuration
# ============================================

# Workspace directory (where data, logs, and output will be stored)
# Matches the rsync destination in sync_to_hpc.txt
WORKSPACE_DIR="/pfs/work9/workspace/scratch/ma_anhnnguy-topic_modeling_data/RAI"

# Script directory (where Python scripts are located - can be same as workspace)
SCRIPT_DIR="$WORKSPACE_DIR"

# Get the mode early for log naming
MODE=${1:-small}

# Change to workspace directory
cd "$WORKSPACE_DIR" || {
    echo "ERROR: Could not change to workspace directory: $WORKSPACE_DIR"
    echo "Please create it first with: mkdir -p $WORKSPACE_DIR"
    exit 1
}

# Create required directories
mkdir -p "$WORKSPACE_DIR/logs" || {
    echo "ERROR: Could not create logs directory"
    exit 1
}
mkdir -p "$WORKSPACE_DIR/results" || {
    echo "ERROR: Could not create results directory"
    exit 1
}

echo "Log files: logs/visa_wall_${SLURM_JOB_ID}.out/err (Mode: $MODE)"

# ============================================
# Print Job Information
# ============================================

echo "=========================================="
echo "THE VISA WALL: LLM BIAS EVALUATION"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPU(s): $SLURM_GPUS_ON_NODE"
echo "CPUs: $SLURM_CPUS_PER_GPU"
echo "Memory: $SLURM_MEM_PER_GPU MB"
echo "Workspace: $WORKSPACE_DIR"
echo "Start Time: $(date)"
echo "=========================================="

# ============================================
# Module Loading
# ============================================

echo ""
echo "Loading modules..."

# Load necessary modules (adjust for your cluster)
module load compiler/gnu/14.2 2>/dev/null || echo "Module compiler/gnu not found, skipping"
module load devel/python/3.11.7-gnu-14.2 2>/dev/null || echo "Module python not found, skipping"
module load devel/cuda/12.8 2>/dev/null || module load devel/cuda/12.1 2>/dev/null || echo "CUDA module not found, skipping"

echo "Modules loaded ✓"

# ============================================
# Virtual Environment
# ============================================

echo ""
echo "Setting up Python environment..."

# Option 1: Use existing venv
if [ -d "$HOME/venv/rai_env" ]; then
    source "$HOME/venv/rai_env/bin/activate"
    echo "Activated venv: rai_env"
elif [ -d "$HOME/venv/visa_wall_env" ]; then
    source "$HOME/venv/visa_wall_env/bin/activate"
    echo "Activated venv: visa_wall_env"
elif [ -d "$WORKSPACE_DIR/venv" ]; then
    source "$WORKSPACE_DIR/venv/bin/activate"
    echo "Activated venv: $WORKSPACE_DIR/venv"
else
    echo "No virtual environment found. Creating one..."
    python -m venv "$WORKSPACE_DIR/venv"
    source "$WORKSPACE_DIR/venv/bin/activate"
    pip install --upgrade pip
    pip install -r "$SCRIPT_DIR/requirements.txt"
    echo "Created and activated new venv"
fi

# Option 2: Conda (uncomment if using conda)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate rai_env

# ============================================
# Environment Variables
# ============================================

# Set CUDA_VISIBLE_DEVICES based on allocated GPUs
# For multi-GPU jobs, expose all allocated GPUs
NUM_GPUS=${SLURM_GPUS_ON_NODE:-1}
if [ "$NUM_GPUS" -ge 4 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
elif [ "$NUM_GPUS" -ge 2 ]; then
    export CUDA_VISIBLE_DEVICES=0,1
else
    export CUDA_VISIBLE_DEVICES=0
fi
export OMP_NUM_THREADS=${SLURM_CPUS_PER_GPU:-12}
export TOKENIZERS_PARALLELISM=false

# ============================================
# HuggingFace Model Cache (PERSISTENT WORKSPACE)
# ============================================
# Use persistent workspace for model downloads to avoid re-downloading every job
# Models are ~200GB total, stored in model_cache workspace

# Find model_cache workspace (created with: ws_allocate -F pfs7wor9 model_cache 14)
MODEL_CACHE_WORKSPACE=$(ws_find model_cache 2>/dev/null)

# Fallback if ws_find fails or workspace doesn't exist
if [ -z "$MODEL_CACHE_WORKSPACE" ] || [ ! -d "$MODEL_CACHE_WORKSPACE" ]; then
    echo "WARNING: model_cache workspace not found!"
    echo "Creating model cache in TMPDIR (will re-download models)"
    echo "To create persistent cache, run: ws_allocate -F pfs7wor9 model_cache 14"
    MODEL_CACHE_WORKSPACE="${TMPDIR:-/tmp}"
fi

# Set HuggingFace cache to persistent workspace
export HF_HOME="${MODEL_CACHE_WORKSPACE}/huggingface_cache"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

# Ensure model cache directory exists
mkdir -p "$HF_HOME/hub"
mkdir -p "$HF_HOME/datasets"

echo ""
echo "Model Cache Configuration:"
echo "  Model cache workspace: $MODEL_CACHE_WORKSPACE"
echo "  HF_HOME: $HF_HOME"
echo "  Models will be cached persistently across jobs"

# ============================================
# FIX: Clear stale PyTorch/vLLM caches from previous jobs
# ============================================
# PyTorch inductor and vLLM cache compiled kernels with absolute paths
# to scratch directories. When a new job starts, these paths are stale
# and cause "Permission denied" errors pointing to old job directories.

echo ""
echo "Clearing stale caches from previous jobs..."

# Clear vLLM cache (contains compiled kernels with stale paths)
if [ -d "$HOME/.cache/vllm" ]; then
    rm -rf "$HOME/.cache/vllm"
    echo "  Cleared: ~/.cache/vllm"
fi

# Clear PyTorch inductor cache
if [ -d "$HOME/.cache/torch_inductor" ]; then
    rm -rf "$HOME/.cache/torch_inductor"
    echo "  Cleared: ~/.cache/torch_inductor"
fi

# Clear Triton cache
if [ -d "$HOME/.cache/triton" ]; then
    rm -rf "$HOME/.cache/triton"
    echo "  Cleared: ~/.cache/triton"
fi

# Clear torch compile cache
if [ -d "$HOME/.cache/torch" ]; then
    rm -rf "$HOME/.cache/torch"
    echo "  Cleared: ~/.cache/torch"
fi

echo "  Cache cleanup complete ✓"

# ============================================
# FIX: Set cache directories to job-local TMPDIR
# ============================================
# Use TMPDIR (job-specific scratch) to avoid cross-job cache conflicts

export TORCH_COMPILE_CACHE_DIR="${TMPDIR:-/tmp}/torch_compile_${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${TMPDIR:-/tmp}/triton_${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/torch_inductor_${SLURM_JOB_ID}"
export TORCH_INDUCTOR_CACHE_DIR="${TMPDIR:-/tmp}/torch_inductor_${SLURM_JOB_ID}"
export TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE=0

# vLLM specific cache settings
export VLLM_CACHE_ROOT="${TMPDIR:-/tmp}/vllm_${SLURM_JOB_ID}"

# Create cache directories
mkdir -p "$TORCH_COMPILE_CACHE_DIR"
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
mkdir -p "$VLLM_CACHE_ROOT"

echo ""
echo "Cache directories configured (job-local in \$TMPDIR):"
echo "  TMPDIR: ${TMPDIR:-/tmp}"
echo "  SLURM_JOB_ID: $SLURM_JOB_ID"
echo "  HF_HOME: $HF_HOME (persistent)"
echo "  TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE (persistent)"
echo "  TORCH_COMPILE_CACHE_DIR: $TORCH_COMPILE_CACHE_DIR (job-local)"
echo "  VLLM_CACHE_ROOT: $VLLM_CACHE_ROOT (job-local)"

# For vLLM memory management
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# ============================================
# Print Environment Info
# ============================================

echo ""
echo "Environment Information:"
echo "  Python version: $(python --version 2>&1)"
echo "  Python path: $(which python)"
echo "  vLLM version: $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo 'Not installed')"
echo "  CUDA version: $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo 'N/A')"
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free,temperature.gpu --format=csv,noheader 2>/dev/null || echo "No GPU info available"
echo "=========================================="

# ============================================
# Path Configuration
# ============================================

PYTHON_SCRIPT="$SCRIPT_DIR/run_bias_evaluation.py"
CONFIG_FILE="$SCRIPT_DIR/config.yaml"
OUTPUT_DIR="$WORKSPACE_DIR/results"

# Verify script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "ERROR: Python script not found at: $PYTHON_SCRIPT"
    echo "Files in directory:"
    ls -la "$SCRIPT_DIR"
    exit 1
fi

# Verify config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found at: $CONFIG_FILE"
    exit 1
fi

# ============================================
# Pre-flight Checks
# ============================================

echo ""
echo "Pre-flight Checks:"
echo "  Python script: $PYTHON_SCRIPT"
echo "  Config file: $CONFIG_FILE"
echo "  Output dir: $OUTPUT_DIR"

# Check if required packages are installed
python -c "import vllm, yaml, pandas" 2>/dev/null || {
    echo ""
    echo "ERROR: Required packages not installed!"
    echo "Installing requirements..."
    pip install -r "$SCRIPT_DIR/requirements.txt"
}

echo "  All packages OK ✓"
echo "=========================================="

# ============================================
# Run Mode Selection
# ============================================
# MODE is already set at the top of the script for log naming
RUNS_PER_CANDIDATE=${RUNS:-5}  # Default 5, override with RUNS env var

echo ""
echo "Run Mode: $MODE"
echo "Runs per candidate: $RUNS_PER_CANDIDATE"
echo "=========================================="

case $MODE in
    "test")
        # Quick test run (1 model, 1 run, no mitigation)
        echo "Running quick test..."
        CMD="python $PYTHON_SCRIPT \
            --config $CONFIG_FILE \
            --models qwen3-4b \
            --runs 1 \
            --output $OUTPUT_DIR/test \
            --no-mitigation \
            --gpu-memory 0.90 \
            --max-model-len 4096"
        ;;

    "small")
        # Small models (4B-14B) - PARALLEL execution on 4 GPUs for 30min time limit
        echo "Running small models (4B-14B) in PARALLEL on 4 GPUs..."
        echo "Total models: 6 (ministral-8b, qwen3-4b, qwen3-8b, llama31-8b, ministral-14b-reasoning, gemma2-9b)"
        echo "Estimated time: ~15-20 minutes (vs ~45 min sequential)"
        echo ""
        
        # Check if we have 4+ GPUs available for parallel execution
        if [ "$NUM_GPUS" -ge 4 ]; then
            echo ">>> Starting PARALLEL execution on $NUM_GPUS GPUs..."
            
            # GPU 0: ministral-8b (~16GB VRAM)
            echo "Starting ministral-8b on GPU 0..."
            CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT \
                --config $CONFIG_FILE \
                --models ministral-8b \
                --runs $RUNS_PER_CANDIDATE \
                --output $OUTPUT_DIR \
                --gpu-memory 0.90 \
                --max-model-len 4096 &
            PID1=$!
            
            # GPU 1: qwen3-4b + qwen3-8b (~24GB VRAM total)
            echo "Starting qwen3-4b,qwen3-8b on GPU 1..."
            CUDA_VISIBLE_DEVICES=1 python $PYTHON_SCRIPT \
                --config $CONFIG_FILE \
                --models qwen3-4b qwen3-8b \
                --runs $RUNS_PER_CANDIDATE \
                --output $OUTPUT_DIR \
                --gpu-memory 0.85 \
                --max-model-len 4096 &
            PID2=$!
            
            # GPU 2: llama31-8b (~16GB VRAM)
            echo "Starting llama31-8b on GPU 2..."
            CUDA_VISIBLE_DEVICES=2 python $PYTHON_SCRIPT \
                --config $CONFIG_FILE \
                --models llama31-8b \
                --runs $RUNS_PER_CANDIDATE \
                --output $OUTPUT_DIR \
                --gpu-memory 0.90 \
                --max-model-len 4096 &
            PID3=$!
            
            # GPU 3: ministral-14b-reasoning (~28GB VRAM)
            echo "Starting ministral-14b-reasoning on GPU 3..."
            CUDA_VISIBLE_DEVICES=3 python $PYTHON_SCRIPT \
                --config $CONFIG_FILE \
                --models ministral-14b-reasoning \
                --runs $RUNS_PER_CANDIDATE \
                --output $OUTPUT_DIR \
                --gpu-memory 0.90 \
                --max-model-len 4096 &
            PID4=$!
            
            # Wait for first parallel batch to complete
            echo ""
            echo ">>> Waiting for parallel batch to complete..."
            echo "  PID $PID1: ministral-8b (GPU 0)"
            echo "  PID $PID2: qwen3-4b,qwen3-8b (GPU 1)"
            echo "  PID $PID3: llama31-8b (GPU 2)"
            echo "  PID $PID4: ministral-14b-reasoning (GPU 3)"
            wait $PID1 $PID2 $PID3 $PID4
            echo ">>> Parallel batch completed!"
            
            # Run Gemma 9B last (needs special handling for softcapping)
            echo ""
            echo ">>> Running gemma2-9b (with eager mode for softcapping support)..."
            CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT \
                --config $CONFIG_FILE \
                --models gemma2-9b \
                --runs $RUNS_PER_CANDIDATE \
                --output $OUTPUT_DIR \
                --gpu-memory 0.90 \
                --max-model-len 4096 \
                --enforce-eager
            
            echo ""
            echo ">>> All small models completed in PARALLEL mode!"
            
        else
            # Fallback to sequential execution if less than 4 GPUs
            echo "WARNING: Only $NUM_GPUS GPUs available. Using sequential execution."
            echo ""
            echo ">>> Running non-Gemma small models (sequential)..."
            python $PYTHON_SCRIPT \
                --config $CONFIG_FILE \
                --models ministral-8b qwen3-4b qwen3-8b llama31-8b ministral-14b-reasoning \
                --runs $RUNS_PER_CANDIDATE \
                --output $OUTPUT_DIR \
                --gpu-memory 0.90 \
                --max-model-len 4096
            
            echo ""
            echo ">>> Running gemma2-9b (with eager mode for softcapping support)..."
            python $PYTHON_SCRIPT \
                --config $CONFIG_FILE \
                --models gemma2-9b \
                --runs $RUNS_PER_CANDIDATE \
                --output $OUTPUT_DIR \
                --gpu-memory 0.90 \
                --max-model-len 4096 \
                --enforce-eager
        fi
        
        CMD=""
        ;;

    "medium")
        # Medium models (24B, 2 GPUs for A100 40GB)
        echo "Running medium models (24B)..."
        CMD="python $PYTHON_SCRIPT \
            --config $CONFIG_FILE \
            --models mistral-small \
            --runs $RUNS_PER_CANDIDATE \
            --output $OUTPUT_DIR \
            --gpu-memory 0.90 \
            --tensor-parallel 2 \
            --max-model-len 4096"
        ;;

    "large")
        # Large models (27B-32B) - PARALLEL execution on 4 GPUs for 30min time limit
        echo "Running large models (27B-32B) in PARALLEL..."
        echo "Total models: 3 (qwen3-30b, qwen3-32b, gemma2-27b)"
        echo "Strategy: 2 models parallel + 1 sequential (Gemma needs special handling)"
        echo "Estimated time: ~20 minutes (vs ~45 min sequential)"
        echo ""
        
        # Check if we have 4+ GPUs for parallel execution
        if [ "$NUM_GPUS" -ge 4 ]; then
            echo ">>> Starting PARALLEL execution on $NUM_GPUS GPUs..."
            
            # Parallel batch: Run 2 large models simultaneously
            # GPU 0-1: qwen3-30b (30B, 2-GPU tensor parallel)
            echo "Starting qwen3-30b on GPU 0-1 (tensor parallel)..."
            CUDA_VISIBLE_DEVICES=0,1 python $PYTHON_SCRIPT \
                --config $CONFIG_FILE \
                --models qwen3-30b \
                --runs $RUNS_PER_CANDIDATE \
                --output $OUTPUT_DIR \
                --gpu-memory 0.85 \
                --tensor-parallel 2 \
                --max-model-len 4096 &
            PID1=$!
            
            # GPU 2-3: qwen3-32b (32B, 2-GPU tensor parallel)
            echo "Starting qwen3-32b on GPU 2-3 (tensor parallel)..."
            CUDA_VISIBLE_DEVICES=2,3 python $PYTHON_SCRIPT \
                --config $CONFIG_FILE \
                --models qwen3-32b \
                --runs $RUNS_PER_CANDIDATE \
                --output $OUTPUT_DIR \
                --gpu-memory 0.85 \
                --tensor-parallel 2 \
                --max-model-len 4096 &
            PID2=$!
            
            # Wait for parallel batch to complete
            echo ""
            echo ">>> Waiting for parallel batch to complete..."
            echo "  PID $PID1: qwen3-30b (GPU 0-1)"
            echo "  PID $PID2: qwen3-32b (GPU 2-3)"
            wait $PID1 $PID2
            echo ">>> Parallel batch completed!"
            
            # Run Gemma 27B last (needs special handling for softcapping)
            echo ""
            echo ">>> Running gemma2-27b (with eager mode for softcapping support)..."
            CUDA_VISIBLE_DEVICES=0,1 python $PYTHON_SCRIPT \
                --config $CONFIG_FILE \
                --models gemma2-27b \
                --runs $RUNS_PER_CANDIDATE \
                --output $OUTPUT_DIR \
                --gpu-memory 0.85 \
                --tensor-parallel 2 \
                --max-model-len 4096 \
                --enforce-eager
            
            echo ""
            echo ">>> All large models completed in PARALLEL mode!"
            
        else
            # Fallback to sequential execution if less than 4 GPUs
            echo "WARNING: Only $NUM_GPUS GPUs available. Using sequential execution."
            TP_SIZE=${SLURM_GPUS_ON_NODE:-2}
            echo "Using $TP_SIZE GPUs for tensor parallelism"
            
            # Run non-Gemma large models first
            echo ""
            echo ">>> Running non-Gemma large models (sequential)..."
            for model in qwen3-30b qwen3-32b; do
                echo ""
                echo ">>> Running $model..."
                python $PYTHON_SCRIPT \
                    --config $CONFIG_FILE \
                    --models $model \
                    --runs $RUNS_PER_CANDIDATE \
                    --output $OUTPUT_DIR \
                    --gpu-memory 0.85 \
                    --tensor-parallel $TP_SIZE \
                    --max-model-len 4096
            done
            
            # Run Gemma 27B with enforce_eager
            echo ""
            echo ">>> Running gemma2-27b (with eager mode for softcapping support)..."
            python $PYTHON_SCRIPT \
                --config $CONFIG_FILE \
                --models gemma2-27b \
                --runs $RUNS_PER_CANDIDATE \
                --output $OUTPUT_DIR \
                --gpu-memory 0.85 \
                --tensor-parallel $TP_SIZE \
                --max-model-len 4096 \
                --enforce-eager
        fi
        
        CMD=""
        ;;

    "all")
        # Run all models sequentially
        echo "Running ALL models..."
        
        # Phase 1: Small models (1 GPU each)
        echo ""
        echo "=========================================="
        echo ">>> Phase 1: Small models (4B-14B, 1 GPU)"
        echo "=========================================="
        for model in ministral-8b qwen3-4b qwen3-8b llama31-8b ministral-14b-reasoning; do
            echo ""
            echo ">>> Running $model..."
            python $PYTHON_SCRIPT \
                --config $CONFIG_FILE \
                --models $model \
                --runs $RUNS_PER_CANDIDATE \
                --output $OUTPUT_DIR \
                --gpu-memory 0.90 \
                --max-model-len 4096
        done
        
        # Gemma 9B with enforce_eager for softcapping
        echo ""
        echo ">>> Running gemma2-9b (with eager mode for softcapping support)..."
        python $PYTHON_SCRIPT \
            --config $CONFIG_FILE \
            --models gemma2-9b \
            --runs $RUNS_PER_CANDIDATE \
            --output $OUTPUT_DIR \
            --gpu-memory 0.90 \
            --max-model-len 4096 \
            --enforce-eager
        
        # Phase 2: Medium/Large models (2 GPUs each)
        echo ""
        echo "=========================================="
        echo ">>> Phase 2: Medium/Large models (24B-32B, 2 GPUs)"
        echo "=========================================="
        for model in mistral-small qwen3-30b qwen3-32b; do
            echo ""
            echo ">>> Running $model..."
            python $PYTHON_SCRIPT \
                --config $CONFIG_FILE \
                --models $model \
                --runs $RUNS_PER_CANDIDATE \
                --output $OUTPUT_DIR \
                --gpu-memory 0.85 \
                --tensor-parallel 2 \
                --max-model-len 4096
        done
        
        # Gemma 27B with enforce_eager for softcapping
        echo ""
        echo ">>> Running gemma2-27b (with eager mode for softcapping support)..."
        python $PYTHON_SCRIPT \
            --config $CONFIG_FILE \
            --models gemma2-27b \
            --runs $RUNS_PER_CANDIDATE \
            --output $OUTPUT_DIR \
            --gpu-memory 0.85 \
            --tensor-parallel 2 \
            --max-model-len 4096 \
            --enforce-eager
        
        # Phase 3: Extra large models (4 GPUs)
        echo ""
        echo "=========================================="
        echo ">>> Phase 3: XLarge models (70B, 4 GPUs)"
        echo "=========================================="
        if [ $(nvidia-smi --list-gpus 2>/dev/null | wc -l) -ge 4 ] || [ "${SLURM_GPUS_ON_NODE:-0}" -ge 4 ]; then
            echo ""
            echo ">>> Running llama31-70b..."
            python $PYTHON_SCRIPT \
                --config $CONFIG_FILE \
                --models llama31-70b \
                --runs $RUNS_PER_CANDIDATE \
                --output $OUTPUT_DIR \
                --gpu-memory 0.90 \
                --tensor-parallel 4 \
                --max-model-len 4096
        else
            echo ""
            echo "WARNING: Skipping Llama-70B - need 4 GPUs"
        fi
        
        # Skip the CMD execution below since we handled it inline
        CMD=""
        ;;

    "qwen")
        # Run QWEN models (4B, 8B, 30B, 32B)
        echo "Running QWEN models..."
        
        # Phase 1: Small QWEN models (4B-8B, 1 GPU each)
        echo ""
        echo "==========================================" 
        echo ">>> Phase 1: Small Qwen models (1 GPU)"
        echo "==========================================" 
        for model in qwen3-4b qwen3-8b; do
            echo ""
            echo ">>> Running $model..."
            python $PYTHON_SCRIPT \
                --config $CONFIG_FILE \
                --models $model \
                --runs $RUNS_PER_CANDIDATE \
                --output $OUTPUT_DIR \
                --gpu-memory 0.90 \
                --max-model-len 4096
        done
        
        # Phase 2: Large QWEN models (30B-32B, 2 GPUs each)
        if [ $(nvidia-smi --list-gpus 2>/dev/null | wc -l) -ge 2 ] || [ "${SLURM_GPUS_ON_NODE:-0}" -ge 2 ]; then
            echo ""
            echo "==========================================" 
            echo ">>> Phase 2: Large Qwen models (2 GPUs)"
            echo "==========================================" 
            for model in qwen3-30b qwen3-32b; do
                echo ""
                echo ">>> Running $model..."
                python $PYTHON_SCRIPT \
                    --config $CONFIG_FILE \
                    --models $model \
                    --runs $RUNS_PER_CANDIDATE \
                    --output $OUTPUT_DIR \
                    --gpu-memory 0.90 \
                    --tensor-parallel 2 \
                    --max-model-len 4096
            done
        else
            echo ""
            echo "WARNING: Skipping Qwen3-30B/32B - need at least 2 GPUs"
            echo "To run 30B+ models, submit with: sbatch --gres=gpu:2 run_experiment.sh qwen"
        fi
        
        # Skip the CMD execution below since we handled it inline
        CMD=""
        ;;

    "gemma")
        # Run Gemma 2 models (9B and 27B) with enforce_eager for softcapping support
        echo "Running Gemma 2 models (with eager mode for softcapping)..."
        
        # Phase 1: Small Gemma model (9B, 1 GPU)
        echo ""
        echo ">>> Phase 1: Gemma-2-9B-IT (9B, 1 GPU, eager mode)"
        python $PYTHON_SCRIPT \
            --config $CONFIG_FILE \
            --models gemma2-9b \
            --runs $RUNS_PER_CANDIDATE \
            --output $OUTPUT_DIR \
            --gpu-memory 0.90 \
            --max-model-len 4096 \
            --enforce-eager
        
        # Phase 2: Large Gemma model (27B, 2 GPUs tensor parallel)
        echo ""
        echo ">>> Phase 2: Gemma-2-27B-IT (27B, 2 GPUs, eager mode)"
        python $PYTHON_SCRIPT \
            --config $CONFIG_FILE \
            --models gemma2-27b \
            --runs $RUNS_PER_CANDIDATE \
            --output $OUTPUT_DIR \
            --gpu-memory 0.85 \
            --tensor-parallel 2 \
            --max-model-len 4096 \
            --enforce-eager
        
        # Skip the CMD execution below since we handled it inline
        CMD=""
        ;;

    "mistral")
        # Run Mistral models (8B, 14B reasoning, 24B)
        echo "Running Mistral models..."
        
        # Phase 1: Small Mistral models (8B-14B, 1 GPU each)
        echo ""
        echo "==========================================" 
        echo ">>> Phase 1: Small Mistral models (1 GPU)"
        echo "==========================================" 
        for model in ministral-8b ministral-14b-reasoning; do
            echo ""
            echo ">>> Running $model..."
            python $PYTHON_SCRIPT \
                --config $CONFIG_FILE \
                --models $model \
                --runs $RUNS_PER_CANDIDATE \
                --output $OUTPUT_DIR \
                --gpu-memory 0.90 \
                --max-model-len 4096
        done
        
        # Phase 2: Medium Mistral model (24B, 2 GPUs recommended)
        echo ""
        echo "==========================================" 
        echo ">>> Phase 2: Mistral-Small-24B (2 GPUs)"
        echo "==========================================" 
        if [ "${SLURM_GPUS_ON_NODE:-1}" -ge 2 ]; then
            python $PYTHON_SCRIPT \
                --config $CONFIG_FILE \
                --models mistral-small \
                --runs $RUNS_PER_CANDIDATE \
                --output $OUTPUT_DIR \
                --gpu-memory 0.90 \
                --tensor-parallel 2 \
                --max-model-len 4096
        else
            echo "WARNING: Running Mistral-Small-24B on 1 GPU (may be slow)"
            python $PYTHON_SCRIPT \
                --config $CONFIG_FILE \
                --models mistral-small \
                --runs $RUNS_PER_CANDIDATE \
                --output $OUTPUT_DIR \
                --gpu-memory 0.95 \
                --max-model-len 4096
        fi
        
        # Skip the CMD execution below since we handled it inline
        CMD=""
        ;;

    "analyze")
        # Run analysis on existing results
        echo "Running analysis on results..."
        echo ""
        
        # Find the most recent results file
        LATEST_RESULT=$(ls -t "$OUTPUT_DIR"/*.json 2>/dev/null | head -1)
        
        if [ -z "$LATEST_RESULT" ]; then
            echo "ERROR: No result files found in $OUTPUT_DIR"
            echo "Run experiments first with: sbatch run_experiment.sh <mode>"
            exit 1
        fi
        
        echo "Analyzing results from: $LATEST_RESULT"
        echo "=========================================="
        python "$SCRIPT_DIR/analyze_results.py" "$LATEST_RESULT"
        
        # Also generate summary for all results if multiple exist
        RESULT_COUNT=$(ls "$OUTPUT_DIR"/*.json 2>/dev/null | wc -l)
        if [ "$RESULT_COUNT" -gt 1 ]; then
            echo ""
            echo "=========================================="
            echo "Found $RESULT_COUNT result files. Analyzing all..."
            echo "=========================================="
            for result_file in "$OUTPUT_DIR"/*.json; do
                echo ""
                echo ">>> Analyzing: $(basename $result_file)"
                python "$SCRIPT_DIR/analyze_results.py" "$result_file"
            done
        fi
        
        CMD=""
        ;;

    "custom")
        # Custom run - specify models via MODELS env variable
        # Usage: MODELS="gemma2-9b ministral-8b" sbatch run_experiment.sh custom
        if [ -z "$MODELS" ]; then
            echo "ERROR: MODELS environment variable not set"
            echo "Usage: MODELS=\"gemma2-9b ministral-8b\" sbatch run_experiment.sh custom"
            exit 1
        fi
        echo "Running custom models: $MODELS"
        echo ""
        
        # Run each model with appropriate GPU configuration
        # This avoids wasting GPU resources by using tensor_parallel=4 for small models
        for model in $MODELS; do
            echo "=========================================="
            echo ">>> Processing model: $model"
            echo "=========================================="
            
            # Determine optimal GPU configuration per model
            case $model in
                llama31-70b)
                    # 70B model needs 4 GPUs
                    TP_SIZE=4
                    GPU_MEM=0.85
                    EAGER_FLAG=""
                    echo "  Config: 70B model - 4 GPUs (tensor_parallel=4)"
                    ;;
                gemma2-27b|qwen3-30b|qwen3-32b|mistral-small)
                    # 27B-32B models need 2 GPUs
                    TP_SIZE=2
                    GPU_MEM=0.85
                    if [[ "$model" =~ "gemma2" ]]; then
                        EAGER_FLAG="--enforce-eager"
                        echo "  Config: 27B model - 2 GPUs (tensor_parallel=2) + eager mode for Gemma"
                    else
                        EAGER_FLAG=""
                        echo "  Config: 27B-32B model - 2 GPUs (tensor_parallel=2)"
                    fi
                    ;;
                gemma2-9b)
                    # Small Gemma needs eager mode but only 1 GPU
                    TP_SIZE=1
                    GPU_MEM=0.90
                    EAGER_FLAG="--enforce-eager"
                    echo "  Config: 9B Gemma - 1 GPU + eager mode for softcapping"
                    ;;
                *)
                    # Small models (4B-14B) use 1 GPU, no eager mode
                    TP_SIZE=1
                    GPU_MEM=0.90
                    EAGER_FLAG=""
                    echo "  Config: Small model - 1 GPU"
                    ;;
            esac
            
            # Build command for this specific model
            MODEL_CMD="python $PYTHON_SCRIPT \
                --config $CONFIG_FILE \
                --models $model \
                --runs ${RUNS:-3} \
                --output $OUTPUT_DIR \
                --gpu-memory $GPU_MEM \
                --tensor-parallel $TP_SIZE \
                --max-model-len 4096 \
                $EAGER_FLAG"
            
            echo ""
            echo "Running: $MODEL_CMD"
            echo ""
            
            eval $MODEL_CMD
            
            EXIT_CODE=$?
            if [ $EXIT_CODE -ne 0 ]; then
                echo "ERROR: Model $model failed with exit code $EXIT_CODE"
                echo "Continuing with next model..."
            else
                echo "✓ Model $model completed successfully"
            fi
            echo ""
        done
        
        # Skip the CMD execution below since we handled it inline
        CMD=""
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo ""
        echo "Available modes:"
        echo "  test    - Quick test (1 model, 1 run)"
        echo "  small   - Small models (3B-12B, 1 GPU)"
        echo "  medium  - Medium models (24B, 1 GPU)"
        echo "  large   - Large models (27B-30B, 2+ GPUs)"
        echo "  all     - Run all models sequentially"
        echo ""
        echo "  === Per-Family Modes (for testing) ==="
        echo "  gemma   - Gemma family (12B + 27B)"
        echo "  qwen    - QWEN family (4B + 30B)"
        echo "  mistral - Mistral family (3B + 24B)"
        echo ""
        echo "  === Utilities ==="
        echo "  analyze - Run analysis on existing results"
        echo "  custom  - Custom models via MODELS env var"
        exit 1
        ;;
esac

# ============================================
# Execute Command
# ============================================

if [ ! -z "$CMD" ]; then
    echo ""
    echo "Executing command:"
    echo "$CMD"
    echo "=========================================="
    
    eval $CMD
fi

# ============================================
# Post-execution
# ============================================

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Job completed successfully!"
    echo "  End Time: $(date)"
    echo "  Results saved to: $OUTPUT_DIR"
    
    # Show result files
    echo ""
    echo "Output Files:"
    ls -lah "$OUTPUT_DIR"/*.json "$OUTPUT_DIR"/*.csv 2>/dev/null || echo "  No result files found"
    
    # Run analysis if results exist
    if ls "$OUTPUT_DIR"/*.json 1>/dev/null 2>&1; then
        echo ""
        echo "Running analysis..."
        python "$SCRIPT_DIR/analyze_results.py" $(ls -t "$OUTPUT_DIR"/*.json | head -1)
    fi
else
    echo "✗ Job failed with exit code: $EXIT_CODE"
    echo "  End Time: $(date)"
    echo ""
    echo "Check logs for details:"
    echo "  $WORKSPACE_DIR/logs/visa_wall_$SLURM_JOB_ID.err"
fi
echo "=========================================="

exit $EXIT_CODE
