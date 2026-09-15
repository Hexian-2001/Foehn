#!/bin/bash
#SBATCH --job-name=daily_forecast
#SBATCH --account=pawsey0115
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/pawsey0115/hwang4/projects/Foehn/logs/daily/daily_%j.out

# Self-resubmitting daily end-to-end forecast (inference + push).
# Submit once (optionally with --begin); each run re-queues the next day.
set -euo pipefail
bash /scratch/pawsey0115/hwang4/projects/Foehn/scripts/daily_run.sh
