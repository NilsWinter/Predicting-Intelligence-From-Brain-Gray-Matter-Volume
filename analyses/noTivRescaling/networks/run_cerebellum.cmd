#!/bin/bash
#
#SBATCH --job-name=cerebellum_noTiv
#SBATCH --output=/scratch/tmp/wintern/iq_frankfurt/analyses/noTivRescaling/networks/cerebellum_noTiv.dat
#SBATCH --mail-type=ALL
#SBATCH --mail-user=wintern@uni-muenster.de
#
#SBATCH --partition normal
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=2-00:00:00
#
# run the application
module add intel
module add Python

# go to working directory
cd /scratch/tmp/wintern/iq_frankfurt/analyses/noTivRescaling/networks/

# activate python environment
source /scratch/tmp/wintern/iq_frankfurt/envs/photonai-env/bin/activate

# finally run the script
python3 networks.py 8