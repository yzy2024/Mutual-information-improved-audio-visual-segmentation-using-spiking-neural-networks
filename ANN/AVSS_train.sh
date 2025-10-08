#!/bin/bash
#SBATCH --job-name=test_gpu_job
#SBATCH --output=/home/songyu/vit-llm/yzy/MI_AVS/work_dir_avss/avss_job_output_imou.log
#SBATCH --error=/home/songyu/vit-llm/yzy/MI_AVS/work_dir_avss/avss_job_error_imou.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH -p gpu
#SBATCH -w NODE10
#SBATCH --gres=gpu:6
export PATH=/home/songyu/conda/envs/AVSegformer/bin
source /home/songyu/conda/etc/profile.d/conda.sh
conda activate AVSegformer
python /home/songyu/vit-llm/yzy/MI_AVS/scripts/avss/train.py 