export PATH=/usr/local/bin:$PATH
export PYTHONPATH=/home/scratch.guyueh_sw/2023su/NeMo:/home/scratch.guyueh_sw/2023su/Megatron-LM-gitlab:$PYTHONPATH

micro_batch=${1:-1}
max_seq=${2:-2048}
freeze_before_training=${3:-"False"}
logfile="gpt5B_nemo_lora"

/home/scratch.svc_compute_arch/release/nsightSystems/x86_64/rel/2023.2.1.122/bin/nsys \
profile -s none -o ./nsys_nemo_lora_profile_batch_${micro_batch}_seq_${max_seq}_freeze_${freeze_before_training} \
-t cuda,nvtx --force-overwrite true \
--capture-range=cudaProfilerApi --capture-range-end=stop \
python /home/scratch.guyueh_sw/2023su/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_peft_tuning.py \
trainer.precision=bf16 \
++model.data.train_ds.pad_to_max_length=True \
model.data.train_ds.max_seq_length=${max_seq} \
trainer.max_epochs=1 \
trainer.val_check_interval=1.0 \
model.global_batch_size=4 \
model.micro_batch_size=${micro_batch} \
model.restore_from_path=/home/scratch.guyueh_sw/2023su/ckpt/nemo_gpt5B \
model.data.train_ds.concat_sampling_probabilities=[1.0] \
model.data.train_ds.file_names=[/home/scratch.guyueh_sw/2023su/dataset/SQuAD/squad_train.jsonl] \
model.data.validation_ds.file_names=[/home/scratch.guyueh_sw/2023su/dataset/SQuAD/squad_val.jsonl] \
model.peft.peft_scheme=lora \
model.answer_only_loss=True \
++model.use_flash_attention=True \
++model.freeze_before_training=${freeze_before_training} \
++model.torch_compile=${torch_compile} \
++model.nsys_profile.enabled=True \
++model.nsys_profile.start_step=2 \
++model.nsys_profile.end_step=2 \
++model.nsys_profile.gen_shape=True \
2>&1 | tee nsys_batch_${micro_batch}_seq_${max_seq}_freeze_${freeze_before_training}_${logfile}.log