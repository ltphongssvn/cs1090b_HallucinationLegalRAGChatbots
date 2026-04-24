#!/bin/bash
#SBATCH --job-name=diag_stage4
#SBATCH --output=logs/diag_stage4_%j.log
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --partition=gpu

cd /shared/home/phl690/cs1090b_HallucinationLegalRAGChatbots
source .venv/bin/activate
.venv/bin/python scripts/diagnose_lepard_cl_mismatch.py --force-stage 4
