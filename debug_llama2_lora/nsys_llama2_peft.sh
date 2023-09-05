NEMO=/lustre/fsw/joc/guyueh/llama2_a100_perf/nemo_jason_mcore_llama
MLM=/lustre/fsw/joc/guyueh/llama2_a100_perf/mlm-github
TE=/lustre/fsw/joc/guyueh/llama2_a100_perf/TransformerEngine
export PYTHONPATH=${NEMO}:${MLM}:${TE}:$PYTHONPATH
MICRO_BATCH_SIZE=${1:-1}
TP=${2:-1}
PP=${3:-1}
SP=${4:-"False"}
NUM_DEVICES=${5:-8}
NUM_NODES=${6:-1}
peft_scheme=${7:-"lora"}
PAD_TO_MAX_LENGTH=${8:-"True"}
MODEL=${9:-"7b"}
MAX_SEQ_LENGTH=${10:-2048}
GLOBAL_BATCH_SIZE=${11:-128}

version=$(git rev-parse HEAD)

tag=${MODEL}_${NUM_NODES}_nodes_${NUM_DEVICES}_devices_TP_${TP}_PP_${PP}_SP_${SP}_MBS_${MICRO_BATCH_SIZE}_GBS_${GLOBAL_BATCH_SIZE}_peft_${peft_scheme}_pad_${PAD_TO_MAX_LENGTH}_v_${version}

## squad
DATASET_DIR=/lustre/fsw/joc/guyueh/data/SQuAD
TRAIN="[${DATASET_DIR}/squad_train.jsonl]"
VALID="[${DATASET_DIR}/squad_val.jsonl]"
VALID_NAMES=null
CKPT_DIR=${NEMO}/debug_llama2_lora/jason_cfgs/${MODEL}

nsys \
profile -s none -o ./llama2_peft_${tag} \
-t cuda,nvtx --force-overwrite true \
--capture-range=cudaProfilerApi --capture-range-end=stop \
torchrun --nproc_per_node=${NUM_DEVICES} ${NEMO}/examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py \
--config-path ${NEMO}/examples/nlp/language_modeling/tuning/conf \
--config-name megatron_gpt_peft_tuning_config  \
++cluster_type=BCP \
trainer.max_steps=10000 \
trainer.precision=bf16 \
trainer.num_nodes=${NUM_NODES} \
trainer.devices=${NUM_DEVICES} \
model.micro_batch_size=${MICRO_BATCH_SIZE} \
model.global_batch_size=${GLOBAL_BATCH_SIZE} \
model.peft.peft_scheme=${peft_scheme} \
model.sequence_parallel=${SP} \
model.tensor_model_parallel_size=${TP} \
model.pipeline_model_parallel_size=${PP} \
model.data.train_ds.file_names=${TRAIN} \
model.data.train_ds.max_seq_length=${MAX_SEQ_LENGTH} \
++model.data.train_ds.pad_to_max_length=${PAD_TO_MAX_LENGTH} \
model.data.train_ds.concat_sampling_probabilities=[1.0] \
model.data.validation_ds.file_names=${VALID} \
model.data.validation_ds.max_seq_length=${MAX_SEQ_LENGTH} \
model.restore_from_path=${CKPT_DIR} \
model.megatron_amp_O2=True \
++model.use_flash_attention=True \
++model.nsys_profile.enabled=True \
++model.nsys_profile.start_step=1 \
++model.nsys_profile.end_step=1 \
2>&1 | tee nsys_llama2_peft_${tag}.log
# trainer.max_steps=6000 \
# trainer.precision=bf16 \
# model.data.train_ds.file_names=[<train dataset path>] \
# model.data.validation_ds.file_names=[<val dataset path>] \
# model.data.test_ds.file_names=[<test dataset path>] \
# model.data.train_ds.concat_sampling_probabilities=[1.0] \
# model.restore_from_path=<base model nemo ckpt> \
# model.pipeline_model_parallel_size=1 \
# model.global_batch_size=32 \