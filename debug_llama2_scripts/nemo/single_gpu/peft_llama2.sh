NEMO=/home/scratch.guyueh_sw/2023su/hf_vs_nemo_on_llama/NeMo
SCRIPT_DIR=${NEMO}/examples/nlp/language_modeling/tuning
CKPT_DIR=/home/scratch.guyueh_sw/2023su/ckpt/llama2

export PYTHONPATH=${NEMO}:${PYTHONPATH}
export PATH=/usr/local/bin:$PATH # for pip

###
### training configurations related to perf
###
PRECISION=16
O2=False
MBS_PER_DEVICE=1
GBS=128
VAL_GBS=128
MAX_SEQ_LENGTH=2048
VAL_CHECK_INTERVAL=100
PAD_TO_MAX_LENGTH=True
OPTIM=fused_adam
# OPTIM="distributed fused adam ++model.optim.bucket_cap_mb=200 ++model.optim.overlap_grad_sync=False ++model.optim.contiguous_grad_buffer=True"
MAX_STEPS=1000
LR="1e-6"
PEFT_SCHEME=lora
FLASH_ATTN=True

###
### script configs
###
config_name=megatron_gpt_peft_tuning_config
script_name=megatron_gpt_peft_tuning.py

###
### pretrained model, tokenizer, and dataset configs
###
RESTORE_FROM_PATH=${CKPT_DIR}/7b.nemo

## squad
DATASET_DIR=/home/scratch.guyueh_sw/2023su/dataset/SQuAD
TRAIN="[${DATASET_DIR}/squad_train.jsonl]"
VALID="[${DATASET_DIR}/squad_val.jsonl]"
VALID_NAMES=null

## rare-finch
# TRAIN="[/preproc_data/tool_generated_sft_datasets/rare-finch/rare-finch_commercial.jsonl]"
# #TRAIN="[/preproc_data/tool_generated_sft_datasets/giga-bison/giga-bison_commercial.shuf.jsonl]"
# VALID="[/preproc_data/scale_ai_data/delivery_2023-04-07-val.jsonl]"
# VALID_NAMES="[scale-ai]"

CONCAT_SAMPLING_PROBS="[1.0]"

TOKENIZER_MODEL=${CKPT_DIR}/d1e3abb2c546424a819a77872c2c8cac_tokenizer.model
TOKENIZER_TOKENIZER_MODEL=${CKPT_DIR}/406edc051f0b4e68abfb770be8fc22a2_tokenizer.model

python ${SCRIPT_DIR}/${script_name} \
--config-path="${NEMO}/examples/nlp/language_modeling/tuning/conf" \
--config-name=${config_name} \
trainer.precision=${PRECISION} \
trainer.num_nodes=1 \
trainer.devices=1 \
trainer.max_steps=${MAX_STEPS} \
trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
++model.use_flash_attention=${FLASH_ATTN} \
model.peft.peft_scheme=${PEFT_SCHEME} \
model.restore_from_path=${RESTORE_FROM_PATH} \
++model.tokenizer.model=${TOKENIZER_MODEL} \
++model.tokenizer.tokenizer_model=${TOKENIZER_TOKENIZER_MODEL} \
++model.nsys_profile.enabled=True \
++model.nsys_profile.start_step=2 \
++model.nsys_profile.end_step=2 \
++model.nsys_profile.gen_shape=True \
model.megatron_amp_O2=${O2} \
model.tensor_model_parallel_size=1 \
model.pipeline_model_parallel_size=1 \
model.optim.name=${OPTIM} \
model.optim.lr=${LR} \
model.answer_only_loss=True \
model.activations_checkpoint_granularity=selective \
model.activations_checkpoint_method=block \
model.activations_checkpoint_num_layers=8 \
++model.data.train_ds.pad_to_max_length=${PAD_TO_MAX_LENGTH} \
model.data.train_ds.max_seq_length=${MAX_SEQ_LENGTH} \
model.data.train_ds.micro_batch_size=${MBS_PER_DEVICE} \
model.data.train_ds.global_batch_size=${GBS} \
model.data.train_ds.file_names=${TRAIN} \
model.data.train_ds.concat_sampling_probabilities=${CONCAT_SAMPLING_PROBS} \
model.data.train_ds.num_workers=0 \
model.data.validation_ds.max_seq_length=${MAX_SEQ_LENGTH} \
model.data.validation_ds.file_names=${VALID} \
model.data.validation_ds.names=${VALID_NAMES} \
model.data.validation_ds.micro_batch_size=${MBS_PER_DEVICE} \
model.data.validation_ds.global_batch_size=${VAL_GBS} \
model.data.validation_ds.write_predictions_to_file=False \
model.data.validation_ds.output_file_path_prefix=/results/predictions \
model.data.validation_ds.num_workers=0 \
model.data.validation_ds.metric.name=loss \
2>&1 | tee peft_llama2.log