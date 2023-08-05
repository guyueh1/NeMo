export PATH=/usr/local/bin:$PATH
export PYTHONPATH=/home/scratch.guyueh_sw/2023su/NeMo:/home/scratch.guyueh_sw/2023su/apex:/home/scratch.guyueh_sw/2023su/lightning/src:$PYTHONPATH

micro_batch=${1:-1}
max_seq=${2:-2048}
addon_args=${3:-""}
logfile=${4:-"gpt1.3B_nemo_lora"}

python /home/scratch.guyueh_sw/2023su/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py \
trainer.precision=bf16 \
++model.megatron_amp_O2=True \
++model.data.train_ds.pad_to_max_length=True \
model.data.train_ds.max_seq_length=${max_seq} \
trainer.max_epochs=1 \
trainer.val_check_interval=1.0 \
model.global_batch_size=4 \
model.micro_batch_size=${micro_batch} \
model.restore_from_path=/home/scratch.guyueh_sw/2023su/ckpt/nemo_gpt1.3B \
model.data.train_ds.concat_sampling_probabilities=[1.0] \
model.data.train_ds.file_names=[/home/scratch.guyueh_sw/2023su/dataset/SQuAD/squad_train.jsonl] \
model.data.validation_ds.file_names=[/home/scratch.guyueh_sw/2023su/dataset/SQuAD/squad_val.jsonl] \
model.peft.peft_scheme=lora \
model.answer_only_loss=True \
$addon_args \
2>&1 | tee batch_${micro_batch}_${logfile}_${addon_args}.log