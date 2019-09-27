#!/bin/bash
#
#SBATCH --job-name=whole_brain
#SBATCH --output=whole_brain.dat
#
#SBATCH --partition normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G
#SBATCH --time=7-00:00:00
#
# run the application
module add intel
module add Python

# go to working directory
cd /scratch/tmp/wintern/iq_frankfurt/analyses/whole_brain/

# activate python environment
source /scratch/tmp/wintern/iq_frankfurt/envs/photonai-env/bin/activate

# finally run the script
python3 whole_brain.py
