import transformers
import peft
import torch

print("torch version:", torch.__version__)
print("transformers version:", transformers.__version__)
print("peft version:", peft.__version__)

"""
torch version: 2.0.1+cu117
transformers version: 4.31.0
peft version: 0.4.0
"""

from transformers import AutoModelForCausalLM, AutoConfig, TrainingArguments, Trainer, TrainerCallback
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import os
from transformers import AutoTokenizer, LlamaTokenizer
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_linear_schedule_with_warmup
from tqdm import tqdm
from datasets import load_dataset

# This script is adapted from https://github.com/huggingface/peft/blob/main/examples/causal_language_modeling/peft_prefix_tuning_clm.ipynb 
# by essentially replacing prefix tuning for Lora 

#######################
## Basic configs
#######################
device = "cuda"
# model_name_or_path = "bigscience/bloomz-7b1"
model_name_or_path = "togethercomputer/LLaMA-2-7B-32K"
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=32, lora_alpha=32, lora_dropout=0.1)

dataset_name = "twitter_complaints"
checkpoint_name = f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}_v1.pt".replace(
    "/", "_"
)
text_column = "Tweet text"
label_column = "text_label"
max_length = 2048
lr = 3e-3
num_epochs = 1
batch_size = 4
gbs = 128
grad_accum_step = gbs // batch_size
max_step_per_epoch = 1000 # for profiling we don't need to iterate the entire dataset

#######################
## Build dataloaders
#######################

dataset = load_dataset("ought/raft", dataset_name)

classes = [k.replace("_", " ") for k in dataset["train"].features["Label"].names]
print(classes)
dataset = dataset.map(
    lambda x: {"text_label": [classes[label] for label in x["Label"]]},
    batched=True,
    num_proc=1,
)
print(dataset)
print(dataset["train"][0])

"""
['Unlabeled', 'complaint', 'no complaint']
DatasetDict({
    train: Dataset({
        features: ['Tweet text', 'ID', 'Label', 'text_label'],
        num_rows: 50
    })
    test: Dataset({
        features: ['Tweet text', 'ID', 'Label', 'text_label'],
        num_rows: 3399
    })
})
{'Tweet text': '@HMRCcustomers No this is my first job', 'ID': 0, 'Label': 2, 'text_label': 'no complaint'}
"""

# data preprocessing
if "llama" in model_name_or_path:
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
target_max_length = max([len(tokenizer(class_label)["input_ids"]) for class_label in classes])
print("target max length", target_max_length)

#######################
## Data preprocessing
#######################

def preprocess_function(examples):
    batch_size = len(examples[text_column])
    inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
    targets = [str(x) for x in examples[label_column]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
        # print(i, sample_input_ids, label_input_ids)
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    # print(model_inputs)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    batch_size=batch_size,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_datasets["test"]
eval_dataset = processed_datasets["test"]

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

print("length of train dataloader: ", len(train_dataloader))
print("length of evaluation dataloader: ", len(eval_dataloader))

"""
length of train dataloader:  50
length of evaluation dataloader:  50
"""

#######################
## Creating model
#######################

# creating model

# model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16)
model_cfg = AutoConfig.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_config(model_cfg)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
print(model.peft_config)

"""
trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06220594176090199
{'default': LoraConfig(peft_type=<PeftType.LORA: 'LORA'>, auto_mapping=None, base_model_name_or_path='decapoda-research/llama-7b-hf', revision=None, task_type=<TaskType.CAUSAL_LM: 'CAUSAL_LM'>, inference_mode=False, r=8, target_modules=['q_proj', 'v_proj'], lora_alpha=32, lora_dropout=0.1, fan_in_fan_out=False, bias='none', modules_to_save=None, init_lora_weights=True, layers_to_transform=None, layers_pattern=None)}
"""

model = model.to(device)


mem = torch.cuda.memory_allocated()
print("Memory allocated", (mem//1024//1024), "MiB")


# optimizer and lr scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

#######################
## Training and Evaluation
#######################

training_arguments = TrainingArguments(
    per_device_train_batch_size=batch_size,
    max_steps=max_step_per_epoch,
    gradient_accumulation_steps=grad_accum_step,
    per_device_eval_batch_size=batch_size,
    report_to=["none"],
    optim="adamw_hf",
    output_dir="lora_from_llama2_twitter_out",
    fp16=True,
    # torch_compile=True,
)

trainer = Trainer(
    model=model,
    args=training_arguments,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    optimizers=(optimizer, lr_scheduler),
)

import time
class TimerMemReporter(TrainerCallback):
    def __init__(self, timer_start_step=1, timer_end_step=2):
        self.step_counter = -1
        self.timer_start_step = timer_start_step
        self.timer_end_step = timer_end_step
    def on_step_begin(self, args, state, control, **kwargs):
        self.step_counter += 1
        if self.step_counter == self.timer_start_step:
            self.start_time = time.time()
        if self.step_counter == (self.timer_end_step+1):
            duration = time.time() - self.start_time
            step_time = duration / (self.timer_end_step+1-self.timer_start_step)
            print("Step_time", step_time)
            peak_mem = torch.cuda.max_memory_allocated()
            print(f"Peak memory: {peak_mem//1024//1024} MiB")

class NvtxProfilerCallback(TrainerCallback):
    def __init__(self, start_step = 1, stop_step = 1):
        self.step_counter = -1
        self.start_step = start_step
        self.stop_step = stop_step
    def on_step_begin(self, args, state, control, **kwargs):
        self.step_counter += 1
        if self.step_counter == self.start_step:
            torch.cuda.cudart().cudaProfilerStart()
        if self.step_counter == self.stop_step + 1:
            torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.nvtx.range_push(f"step_{self.step_counter}")
    
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.nvtx.range_pop()

trainer.add_callback(TimerMemReporter())
trainer.add_callback(NvtxProfilerCallback())
trainer.train()

# # training loop
# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0
#     for step, batch in enumerate(tqdm(train_dataloader)):
#         if step >= max_step_per_epoch:
#             break
#         batch = {k: v.to(device) for k, v in batch.items()}
#         #         print(batch)
#         #         print(batch["input_ids"].shape)
#         outputs = model(**batch)
#         loss = outputs.loss
#         total_loss += loss.detach().float()
#         loss.backward()
#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()

#     # model.eval()
#     # eval_loss = 0
#     # eval_preds = []
#     # for step, batch in enumerate(tqdm(eval_dataloader)):
#     #     batch = {k: v.to(device) for k, v in batch.items()}
#     #     with torch.no_grad():
#     #         outputs = model(**batch)
#     #     loss = outputs.loss
#     #     eval_loss += loss.detach().float()
#     #     eval_preds.extend(
#     #         tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
#     #     )

#     # eval_epoch_loss = eval_loss / len(eval_dataloader)
#     # eval_ppl = torch.exp(eval_epoch_loss)
#     # train_epoch_loss = total_loss / len(train_dataloader)
#     # train_ppl = torch.exp(train_epoch_loss)
#     # print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

#######################
## Export model
#######################

peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
model.save_pretrained(peft_model_id)

