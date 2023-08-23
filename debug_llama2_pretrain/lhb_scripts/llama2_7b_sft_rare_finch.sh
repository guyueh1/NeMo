#!/bin/bash
#SBATCH -N 4 --ntasks-per-node 8 -A coreai_devtech_all -p luna --job-name coreai_devtech_all-gpt:gpt:sft_7b_mcore -t 120

<<"COMMENT"
#SBATCH -A devtech
#SBATCH -p luna
#SBATCH -N 1
#SBATCH -t 4:00:00
#SBATCH -J "devtech-DLFW:sft"
#SBATCH --ntasks-per-node=8
COMMENT

set -x

#CONTAINER="nvcr.io/ea-bignlp/bignlp-training:23.04-py3" # use own pre-built nemo?
#CONTAINER="nvcr.io/ea-bignlp/nemofw-training:23.07-py3"
CONTAINER="gitlab-master.nvidia.com/hongbinl/external_images:nemofw-23.07-ptl-2.0"
WANDB="55b1e23693852385f4eae5b38c3c2bbcda88b585" 

# Model config: conf/megatron_gpt_config.yaml
CONFIG_PATH='conf'
CONFIG_NAME='megatron_gpt_sft'

GLOBAL_BATCH_SIZE=128
VALID_GLOBAL_BATCH_SIZE=128
MICRO_BATCH_SIZE=1
ACCUMULATE_GRAD_BATCHES=1
TENSOR_MODEL_PARALLEL_SIZE=1
PIPELINE_MODEL_PARALLEL_SIZE=1
VAL_CHECK_INTERVAL=100
MAX_STEPS=1000
DATA_SPLITS_STRING="\'99982,9,9\'"
LR="1e-6"

# Model architecture
MAX_SEQ_LENGTH=4096

# Logging
PROJECT="nemo_llama2_mcore_sft"
EXPNAME="llama2_7b_tp2_mcore"

# Mounts
GPFS="/lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron/nemo_repo/internal/NeMo"
PREPROC_DATA="/lustre/fsw/swdl/swdl-langspeech/datasets/data/BigNLP/"
#MEGATRON_PATH="/lustre/fsw/devtech/hpc-devtech/lit/software/Megatron-LM/"
RESULTS="/lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron/scripts/support_llama/sft/results/${EXPNAME}"

CODE="${GPFS}"
#MODEL="/lustre/fsw/swdl/swdl-langspeech/sandeepsub/models"
MODEL_DIR="/lustre/fsw/joc/big_nlp/nemo_gpt3/rlhf/debug/checkpoints"
MODEL_DIR="/lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron/checkpoints/llama2/Llama-2-7b-hf"
MODEL_NAME="llama2-7b-mcore-tp2"

mkdir -p ${RESULTS}

MOUNTS="--container-mounts=/lustre/fsw:/lustre/fsw"

TRAIN="[${PREPROC_DATA}/tool_generated_sft_datasets/rare-finch/rare-finch_commercial.jsonl]"
#TRAIN="[/lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron/scripts/support_llama/NeMo-Megatron-Launcher/launcher_scripts/data/prompt_data/v1.1/squad_train.jsonl]"
#TRAIN="[/preproc_data/tool_generated_sft_datasets/giga-bison/giga-bison_commercial.shuf.jsonl]"

VALID="[${PREPROC_DATA}/scale_ai_data/delivery_2023-04-07-val.jsonl]"
#VALID="[/lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron/scripts/support_llama/NeMo-Megatron-Launcher/launcher_scripts/data/prompt_data/v1.1/squad_val.jsonl]"

VALID_NAMES="[scale-ai]"
#VALID_NAMES="[squad]"

CONCAT_SAMPLING_PROBS="[1.0]"

# Necessary Exports
export HYDRA_FULL_ERROR=1

OUTFILE="${RESULTS}/slurm-%j-%n.out"
ERRFILE="${RESULTS}/error-%j-%n.out"

APEX="/lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron/apex"
TE="/lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron/nemo_repo/TransformerEngine"
MLM="/lustre/fsw/devtech/hpc-devtech/hongbinl/nemo_megatron/Megatron-LM"

#&& git rev-parse HEAD \
read -r -d '' cmd <<EOF
echo "*******STARTING********" \
&& echo "---------------" \
&& export WANDB_API_KEY=${WANDB} \
&& echo "Starting training" \
&& export PYTHONPATH="${APEX}:${TE}:${MLM}:${CODE}:${PYTHONPATH}" \
&& python ${CODE}/examples/nlp/language_modeling/tuning/megatron_gpt_sft.py \
	--config-path=${CODE}/examples/nlp/language_modeling/tuning/conf \
	--config-name=${CONFIG_NAME} \
	+trainer.limit_val_batches=4 \
	trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
	trainer.devices=${SLURM_NTASKS_PER_NODE} \
	trainer.max_epochs=null \
	trainer.max_steps=${MAX_STEPS} \
	trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
	trainer.precision=bf16 \
	model.megatron_amp_O2=True \
	model.restore_from_path=${MODEL_DIR}/${MODEL_NAME} \
	model.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
	model.pipeline_model_parallel_size=${PIPELINE_MODEL_PARALLEL_SIZE} \
	model.optim.name=distributed_fused_adam \
	model.optim.lr=${LR} \
	++model.optim.bucket_cap_mb=200 \
    	++model.optim.overlap_grad_sync=False \
    	++model.optim.contiguous_grad_buffer=True \
	model.answer_only_loss=True \
	model.sequence_parallel=False \
	model.activations_checkpoint_granularity=selective \
	model.activations_checkpoint_method=block \
	model.activations_checkpoint_num_layers=1 \
	model.data.train_ds.max_seq_length=${MAX_SEQ_LENGTH} \
	model.data.train_ds.micro_batch_size=${MICRO_BATCH_SIZE} \
	model.data.train_ds.global_batch_size=${GLOBAL_BATCH_SIZE} \
	model.data.train_ds.file_names=${TRAIN} \
	model.data.train_ds.concat_sampling_probabilities=${CONCAT_SAMPLING_PROBS} \
	model.data.train_ds.num_workers=0 \
	model.data.validation_ds.max_seq_length=${MAX_SEQ_LENGTH} \
	model.data.validation_ds.file_names=${VALID} \
	model.data.validation_ds.names=${VALID_NAMES} \
	model.data.validation_ds.micro_batch_size=${MICRO_BATCH_SIZE} \
	model.data.validation_ds.global_batch_size=${VALID_GLOBAL_BATCH_SIZE} \
	model.data.validation_ds.write_predictions_to_file=False \
	model.data.validation_ds.output_file_path_prefix=${RESULTS}/predictions \
	model.data.validation_ds.num_workers=0 \
	model.data.validation_ds.metric.name=loss \
	model.data.test_ds.max_seq_length=${MAX_SEQ_LENGTH} \
      	model.data.test_ds.file_names=${VALID} \
	model.data.test_ds.names=${VALID_NAMES} \
	model.data.test_ds.micro_batch_size=${MICRO_BATCH_SIZE} \
	model.data.test_ds.global_batch_size=${VALID_GLOBAL_BATCH_SIZE} \
	model.data.test_ds.write_predictions_to_file=False \
	model.data.test_ds.output_file_path_prefix=${RESULTS}/predictions \
	model.data.test_ds.num_workers=0 \
	model.data.test_ds.metric.name=loss \
	exp_manager.create_wandb_logger=True \
	exp_manager.wandb_logger_kwargs.name=${EXPNAME} \
	exp_manager.wandb_logger_kwargs.project=${PROJECT} \
	exp_manager.explicit_log_dir=${RESULTS} \
	exp_manager.resume_if_exists=False \
	exp_manager.resume_ignore_no_checkpoint=True \
	exp_manager.create_checkpoint_callback=True \
	exp_manager.checkpoint_callback_params.monitor=validation_loss \
	++exp_manager.checkpoint_callback_params.save_top_k=3 \
	exp_manager.checkpoint_callback_params.mode=min \
	++exp_manager.max_time_per_run=00:01:45:00 \
	++exp_manager.checkpoint_callback_params.save_nemo_on_train_end=True
EOF

srun --no-container-mount-home --container-image="$CONTAINER" $MOUNTS bash -c "${cmd}"
