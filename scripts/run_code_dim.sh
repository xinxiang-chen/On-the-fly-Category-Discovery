#!/bin/bash
# Run one SMILE config of the code_dim sweep without Slurm.
#
# Usage (from the SMILE repo root):
#   bash scripts/run_code_dim.sh [dataset] [code_dim] [gpu] [seed]
#   bash scripts/run_code_dim.sh cub 256 0          # defaults: cub 256 0 0
#   nohup bash scripts/run_code_dim.sh pets 1024 1 > /dev/null 2>&1 &
#
# Data root holds cub/, oxford_pets/ and dino/dino_vitbase16_pretrain.pth (see config.py):
#   export SMILE_DATA_ROOT=/path/to/Datasets

set -euo pipefail

data_set=${1:-cub}
L=${2:-256}
gpu=${3:-0}
seed=${4:-0}

export SMILE_DATA_ROOT=${SMILE_DATA_ROOT:-$HOME/Datasets}
export CUDA_VISIBLE_DEVICES=$gpu
PYTHON=${PYTHON:-python}

cd "$(dirname "$0")/.."
exp_root=exp/sweep_code_dim/$data_set/"${data_set}_L${L}_seed${seed}"
log_file=log/"${data_set}_L${L}_seed${seed}.log"
mkdir -p log

echo "data_set=${data_set} L=${L} seed=${seed} gpu=${gpu} node=$(hostname)"
echo "exp_root=${exp_root}  SMILE_DATA_ROOT=${SMILE_DATA_ROOT}  log=${log_file}"

# Hyperparameters from train.sh
${PYTHON} -u -m train_smile \
    --dataset_name $data_set \
    --code_dim $L \
    --seeds $seed \
    --exp_root $exp_root \
    --batch_size 128 \
    --grad_from_block 11 \
    --epochs 100 \
    --base_model vit_dino \
    --num_workers 8 \
    --use_ssb_splits 'True' \
    --sup_con_weight 0.5 \
    --weight_decay 5e-5 \
    --contrast_unlabel_only 'False' \
    --transform 'imagenet' \
    --lr 0.01 \
    --eval_funcs 'v1' 'v2' \
    2>&1 | tee "$log_file"
