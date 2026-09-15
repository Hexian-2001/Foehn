#!/bin/bash
#SBATCH --job-name=backfill_forecast
#SBATCH --account=pawsey0115
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/pawsey0115/hwang4/projects/Foehn/logs/backfill/backfill_%j.out

# Backfill missed forecast days: inference (gpu-dev) + push, both models.
set -euo pipefail
bash /scratch/pawsey0115/hwang4/projects/Foehn/scripts/backfill.sh
