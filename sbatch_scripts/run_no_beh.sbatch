#!/bin/bash
#SBATCH --job-name=sen-baseline
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=grete:shared    # see https://www.hlrn.de/doc/display/PUB/Compute+node+partitions for a list of accessible hlrn partitions
#SBATCH -G A100:1                   # number of gpus per node
#SBATCH --time 0-48:00
#SBATCH --constraint "inet" 
#SBATCH --mem=100G
#SBATCH -C 80gb_vram
#SBATCH --output=../slurm_files/slurm-%x-%j.out     # where to write output, %x give job name, %j names job id
#SBATCH --error=../slurm_files/slurm-%x-%j.err      # where to write slurm error
#SBATCH --mail-type=begin            # send mail when job begins
#SBATCH --mail-type=end              # send mail when job ends
#SBATCH --mail-user=goirik.chakrabarty@uni-goettingen.de
export HTTP_PROXY="http://www-cache.gwdg.de:3128"
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export FTP_PROXY="http://www-cache.gwdg.de:3128"

module load apptainer
module load miniforge3
module load git

source /sw/tools/python/anaconda3/2020.11/skl/bin/activate 
conda info --envs
conda activate /mnt/vast-nhr/projects/nix00014/goirik/.conda/experanto
cd /mnt/vast-nhr/projects/nix00014/goirik/MOZAIK-new/latent_space_model

# Printing out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.
python --version
python -m torch.utils.collect_env
#nvcc -V
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python training_scripts/single_session_train_bs8_no_beh.py
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python training_scripts/single_session_train_mozaik_no_beh.py