export PATH=/usr/local/bin:$PATH
export PYTHONPATH=/home/scratch.guyueh_sw/2023su/Nemo:$PYTHONPATH

python examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py \
trainer.precision=bf16 \
++model.megatron_amp_O2=True \
model.restore_from_path=/home/scratch.guyueh_sw/2023su/ckpt/nemo_gpt5B_bf16 \
model.data.train_ds.file_names=[/home/scratch.guyueh_sw/2023su/dataset/SQuAD/squad_train.jsonl] \
model.data.train_ds.concat_sampling_probabilities=[1.0] \
model.data.validation_ds.file_names=[/home/scratch.guyueh_sw/2023su/dataset/SQuAD/squad_val.jsonl] \
model.peft.peft_scheme=lora \
model.answer_only_loss=True \
2>&1 | tee peft_tuning.log