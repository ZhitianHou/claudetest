#!/bin/bash
#SBATCH --job-name=pathology
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1024G
#SBATCH --time=192:00:00
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --output=/work/projects/polyullm/houzht/logs/pathology/local_model_%j_%t_%N.out
#SBATCH --error=/work/projects/polyullm/houzht/logs/pathology/local_model_%j_%t_%N.err
#SBATCH --exclude=kb3-a1-nv-dgx01,kb3-a1-nv-dgx02,kb3-a1-nv-dgx03,kb3-a1-nv-dgx04,kb3-a1-nv-dgx05,kb3-a1-nv-dgx06,kb3-a1-nv-dgx07,kb3-a1-nv-dgx16

# set -x

# replace these information with your own
workdir=/work/projects/polyullm/houzht/PrePath/api_test
container_image=/lustre/projects/polyullm/container/verl-sglang+0503.sqsh
container_name=verl-sglang+0503
container_mounts=/lustre/projects/polyullm:/lustre/projects/polyullm,/work/projects/polyullm:/work/projects/polyullm,/work/projects/polyullm/houzht/miniconda3/envs/medevalkit_qwen3:/zju_0038/medical/miniconda3/envs/medevalkit_qwen3,/work/projects/polyullm:/home/projects/polyullm

# replace these information with your own
NNODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
NPROC_PER_NODE=8
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "Number of nodes: $NNODES"

CONDA_ENV_DIR=medevalkit_qwen35

# output settings
MODEL_PATH=/work/projects/polyullm/houzht/VeOmni-Internal-medical2.0/qwen3_5_27b_sft_npc/global_step_50
DATA_PATH=/work/projects/polyullm/houzht/PrePath/api_test/new_results/data_test2_343.csv # /work/projects/polyullm/houzht/PrePath/api_test/new_results/data_128.csv  # /work/projects/polyullm/houzht/PrePath/api_test/data.csv # /work/projects/polyullm/houzht/PrePath/api_test/data_1202.csv
OUTPUT_PATH=/work/projects/polyullm/houzht/PrePath/api_test/new_results/results_qwen3_5_27b_v2_test2.jsonl  # /work/projects/polyullm/houzht/PrePath/api_test/results_qwen3_5_27b_dnn_sft.jsonl # /work/projects/polyullm/houzht/PrePath/api_test/results_qwen3_5_27b_dnn_sft_1202.jsonl
DNN_RESULTS=/work/projects/polyullm/houzht/PrePath/api_test/new_results/threshold0_5_results_test2.csv  # /work/projects/polyullm/houzht/PrePath/api_test/threshold0.5_results_dnn2.csv # /work/projects/polyullm/houzht/PrePath/api_test/1202_threshold0.5_results.csv
USE_ASYNC="true"
USE_FEWSHOT="false"
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
NUM_SAMPLES=350
SPLIT="false"
MODEL_TYPE="qwen3_5"
MAX_CONCURRENCY=256
ENABLE_THINKING="true"
TP_SIZE=1
DP_SIZE=8

echo "Starting training..."

SCRIPTS="

eval \"\$(/work/projects/polyullm/houzht/miniconda3/bin/conda shell.bash hook)\"

conda activate $CONDA_ENV_DIR

cd /work/projects/polyullm/houzht/PrePath/api_test

export SGLANG_ENABLE_JIT_DEEPGEMM=0
export SGLANG_DISABLE_TRITON=1

python api_test.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --use_async "$USE_ASYNC" \
    --use_fewshot "$USE_FEWSHOT" \
    --dnn_results "$DNN_RESULTS" \
    --model_type "$MODEL_TYPE" \
    --cuda_visible_devices "$CUDA_VISIBLE_DEVICES" \
    --split "$SPLIT" \
    --max_concurrency "$MAX_CONCURRENCY" \
    --enable_thinking "$ENABLE_THINKING" \
    --tensor_parallel_size "$TP_SIZE" \
    --data_parallel_size "$DP_SIZE" \
    --num_samples "$NUM_SAMPLES" 2>&1 | tee results_qwen3_5_27b_v2_test2.txt
"

PYTHONUNBUFFERED=1 srun --overlap \
    --container-name=$container_name \
    --container-mounts=$container_mounts \
    --container-image=$container_image \
    --container-workdir=$workdir \
    --container-writable \
    --container-remap-root \
    bash -c "$SCRIPTS"

