#!/bin/bash
#
#SBATCH --job-name=whole_brain_noTiv
#SBATCH --output=whole_brain_noTiv.dat
#
#SBATCH --partition normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=7-00:00:00
#
# run the application
module add intel
module add Python

# go to working directory
cd /scratch/tmp/wintern/iq_frankfurt/analyses/noTivRescaling/whole_brain/

# activate python environment
source /scratch/tmp/wintern/iq_frankfurt/envs/photonai-env/bin/activate

# finally run the script
python3 whole_brain.py
