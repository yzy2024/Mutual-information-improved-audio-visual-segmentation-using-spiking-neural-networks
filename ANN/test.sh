#!/bin/bash
#SBATCH --job-name=test_gpu_job
#SBATCH --output=/home/songyu/vit-llm/yzy/MI_AVS/work_dir/job_output_imou_test.log
#SBATCH --error=/home/songyu/vit-llm/yzy/MI_AVS/work_dir/job_error_imou_test.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH -p gpu
#SBATCH -w NODE09
#SBATCH --gres=gpu:1
export PATH=/home/songyu/conda/envs/AVSegformer/bin
source /home/songyu/conda/etc/profile.d/conda.sh
conda activate AVSegformer
python /home/songyu/vit-llm/yzy/MI_AVS/scripts/s4/test.py