#!/bin/bash
#SBATCH -n 32               # Number of cores (should also specify -N?)
#SBATCH -t 0-4          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p test  # Partition to submit to
#SBATCH --mem=8000           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o cannon_out/myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e cannon_out/myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=ALL
#SBATCH --mail-user=atsang@g.harvard.edu



export XDG_RUNTIME_DIR=/n/scratchlfs/dvorkin_lab/atsang/tmp
echo "XDG_RUNTIME_DIR is $XDG_RUNTIME_DIR"

module load Anaconda3/5.0.1-fasrc02
#source activate jup3
source venv/bin/activate
#which pip
#pip --version
date
python -E effective_convergence.py
date

echo "DONE"
