#!/bin/bash
#SBATCH --job-name=shell_detectron2
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=${SLURM_EMAIL:-${USER}@ufl.edu}

module purge
module load detectron2

srun python train.py \
  --annotation-json data/shell_mixed/train/_annotations.coco.json \
  --image-root      data/shell_mixed/train \
  --val-annotation-json data/shell_mixed/val/_annotations.coco.json \
  --val-image-root      data/shell_mixed/val \
  --output-dir      Detectron2_Models \
  --num-workers     $SLURM_CPUS_PER_TASK \
  --num-gpus        1
