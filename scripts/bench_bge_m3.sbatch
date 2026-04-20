#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/bench_%j.log
set -e
cd "$SLURM_SUBMIT_DIR"
export HF_HOME="/shared/home/$USER/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name --format=csv,noheader
uv run --locked python scripts/bench_bge_m3.py
