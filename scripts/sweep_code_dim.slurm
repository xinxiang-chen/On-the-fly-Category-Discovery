#!/bin/bash
# Hash code length sweep for SMILE on CUB and Pets (CARC Discovery, Slurm job array).
# 8 tasks = 2 datasets x 4 code lengths, one GPU each.
#
# Submit from the SMILE repo root:
#   mkdir -p log/slurm
#   sbatch --account=<project_id> scripts/sweep_code_dim.slurm
# Run a subset, e.g. only Pets:
#   sbatch --account=<project_id> --array=4-7 scripts/sweep_code_dim.slurm
#
#SBATCH --job-name=smile_codedim
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=a40:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --output=log/slurm/%x_%A_%a.out
#SBATCH --error=log/slurm/%x_%A_%a.out

set -euo pipefail

DATASETS=(cub pets)
LENGTHS=(128 256 512 1024)
SEED=0               # first of SMILE's default runs (seeds 0-9)

idx=${SLURM_ARRAY_TASK_ID}
data_set=${DATASETS[$(( idx / ${#LENGTHS[@]} ))]}
L=${LENGTHS[$(( idx % ${#LENGTHS[@]} ))]}

# Directory holding cub/, oxford_pets/ and dino/dino_vitbase16_pretrain.pth (see config.py)
export SMILE_DATA_ROOT=${SMILE_DATA_ROOT:-$HOME/Datasets}

module purge
module load conda
eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV:-phe}

cd "${SLURM_SUBMIT_DIR}"
exp_root=exp/sweep_code_dim/$data_set/"${data_set}_L${L}_seed${SEED}"

echo "task=${idx} data_set=${data_set} L=${L} seed=${SEED} node=$(hostname)"
echo "exp_root=${exp_root}  SMILE_DATA_ROOT=${SMILE_DATA_ROOT}"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Hyperparameters from train.sh
python -m train_smile \
    --dataset_name $data_set \
    --code_dim $L \
    --seeds $SEED \
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
    --eval_funcs 'v1' 'v2'
