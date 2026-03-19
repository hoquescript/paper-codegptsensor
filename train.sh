#!/bin/bash
#SBATCH --job-name=codegptsensor
#SBATCH --partition=gpubase_bygpu_b5
#SBATCH --array=0-1
#SBATCH --time=48:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p "$ROOT_DIR/logs"
cd "$ROOT_DIR"

module load python/3.14

virtualenv --clear "$SLURM_TMPDIR/ENV"
source "$SLURM_TMPDIR/ENV/bin/activate"

pip install --no-index --upgrade pip
pip install --no-index --no-cache-dir \
  numpy \
  pandas \
  torch \
  "transformers==4.57.6" \
  scikit-learn \
  scipy \
  sentencepiece \
  "tree-sitter==0.23.2" \
  "tree-sitter-cpp==0.23.4" \
  "tree-sitter-java==0.23.5" \
  "tree-sitter-python==0.23.6"

export HF_HOME="$SCRATCH/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

python -c "import tree_sitter, tree_sitter_cpp, tree_sitter_python, tree_sitter_java"
python -c "import torch; print(f'torch={torch.__version__} cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')"

LANGUAGES=(
  "java"
  "python"
)
REPRESENTATIONS=(
  "code"
  "ast"
  "combined"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$TASK_ID" -lt 0 ] || [ "$TASK_ID" -ge "${#LANGUAGES[@]}" ]; then
  echo "Invalid SLURM_ARRAY_TASK_ID=$TASK_ID" >&2
  exit 1
fi

LANGUAGE="${LANGUAGES[$TASK_ID]}"

MODEL_NAME="${MODEL_NAME:-microsoft/unixcoder-base-nine}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
BLOCK_SIZE="${BLOCK_SIZE:-400}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
SEED="${SEED:-99}"
DATASET_SUFFIX="${DATASET_SUFFIX:-}"

TRAIN_DATA_FILE="$ROOT_DIR/dataset/$LANGUAGE/train${DATASET_SUFFIX}.jsonl"
EVAL_DATA_FILE="$ROOT_DIR/dataset/$LANGUAGE/valid${DATASET_SUFFIX}.jsonl"
TEST_DATA_FILE="$ROOT_DIR/dataset/$LANGUAGE/test${DATASET_SUFFIX}.jsonl"

for path in "$TRAIN_DATA_FILE" "$EVAL_DATA_FILE" "$TEST_DATA_FILE"; do
  if [ ! -f "$path" ]; then
    echo "Dataset file not found: $path" >&2
    exit 1
  fi
done

echo "Running language=$LANGUAGE"
echo "Train file: $TRAIN_DATA_FILE"
echo "Eval file:  $EVAL_DATA_FILE"
echo "Test file:  $TEST_DATA_FILE"

for REPRESENTATION in "${REPRESENTATIONS[@]}"; do
  OUTPUT_DIR="$ROOT_DIR/CodeGPTSensor/models_output/${LANGUAGE}_${REPRESENTATION}${DATASET_SUFFIX}"

  echo "Running representation=$REPRESENTATION"
  echo "Output dir: $OUTPUT_DIR"

  python "$ROOT_DIR/CodeGPTSensor/run.py" \
    --do_train \
    --representation "$REPRESENTATION" \
    --model_name_or_path "$MODEL_NAME" \
    --train_data_file "$TRAIN_DATA_FILE" \
    --eval_data_file "$EVAL_DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_EPOCHS" \
    --block_size "$BLOCK_SIZE" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --seed "$SEED" \
    --contrast

  python "$ROOT_DIR/CodeGPTSensor/run.py" \
    --do_test \
    --representation "$REPRESENTATION" \
    --model_name_or_path "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --test_data_file "$TEST_DATA_FILE" \
    --block_size "$BLOCK_SIZE" \
    --eval_batch_size "$EVAL_BATCH_SIZE" \
    --seed "$SEED"
done
