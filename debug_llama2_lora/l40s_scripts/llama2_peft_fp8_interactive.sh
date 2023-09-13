NEMO=/home/guyueh/llama2_l40s_profile/guyueh1_nemo
export PYTHONPATH=${NEMO}:$PYTHONPATH
MICRO_BATCH_SIZE=${1:-1}
TP=${2:-1}
SP=${3:-"False"}
NUM_DEVICES=${4:-8}
NUM_NODES=${5:-1}
peft_scheme=${6:-"lora"}
PAD_TO_MAX_LENGTH=${7:-"True"}

version=$(git rev-parse HEAD)

tag=${NUM_NODES}_nodes_${NUM_DEVICES}_devices_TP_${TP}_SP_${SP}_MBS_${MICRO_BATCH_SIZE}_peft_${peft_scheme}_pad_${PAD_TO_MAX_LENGTH}_v_${version}

## squad
DATASET_DIR=/home/guyueh/llama2_l40s_profile/data/SQuAD
TRAIN="[${DATASET_DIR}/squad_train.jsonl]"
VALID="[${DATASET_DIR}/squad_val.jsonl]"
VALID_NAMES=null
CKPT_DIR=${NEMO}/debug_llama2_lora/l40s_scripts/jason_cfgs/7b

NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=0 \
torchrun --nproc_per_node=${NUM_DEVICES} ${NEMO}/examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py \
--config-path=${NEMO}/examples/nlp/language_modeling/tuning/conf \
--config-name megatron_gpt_peft_tuning_config  \
++cluster_type=BCP \
trainer.max_steps=10000 \
trainer.precision=bf16 \
trainer.num_nodes=${NUM_NODES} \
trainer.devices=${NUM_DEVICES} \
model.micro_batch_size=${MICRO_BATCH_SIZE} \
model.peft.peft_scheme=${peft_scheme} \
model.sequence_parallel=${SP} \
model.tensor_model_parallel_size=${TP} \
model.pipeline_model_parallel_size=1 \
model.data.train_ds.file_names=${TRAIN} \
model.data.validation_ds.file_names=${VALID} \
++model.data.train_ds.pad_to_max_length=${PAD_TO_MAX_LENGTH} \
model.data.train_ds.concat_sampling_probabilities=[1.0] \
model.restore_from_path=${CKPT_DIR} \
model.megatron_amp_O2=True \
++model.fp8=True \
++model.fp8_e4m3=True \
2>&1 | tee llama2_peft_${tag}.log
# trainer.max_steps=6000 \
# trainer.precision=bf16 \
# model.data.train_ds.file_names=[<train dataset path>] \
# model.data.validation_ds.file_names=[<val dataset path>] \
# model.data.test_ds.file_names=[<test dataset path>] \
# model.data.train_ds.concat_sampling_probabilities=[1.0] \
# model.restore_from_path=<base model nemo ckpt> \
# model.pipeline_model_parallel_size=1 \
# model.global_batch_size=32 \