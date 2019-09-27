#!/bin/bash
#
#SBATCH --job-name=perm_whole_brain
#SBATCH --output=output_perm_whole_brain.dat

#
#SBATCH --partition normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=7-00:00:00
#
#SBATCH --array=1-1000%50
#
# run the application
module add intel
module add Python

# go to working directory
cd /scratch/tmp/wintern/iq_frankfurt/analyses/whole_brain/

# activate python environment
source /scratch/tmp/wintern/iq_frankfurt/envs/photonai-env/bin/activate

# extract row for task
row=$(sed -n $SLURM_ARRAY_TASK_ID'p' /scratch/tmp/wintern/iq_frankfurt/data/iq_prediction_perms_rs_15.csv)

# finally run the script
python3 perm_whole_brain.py $row