import transformers
import peft
import torch
from accelerate import Accelerator

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
batch_size = 1
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

accelerator = Accelerator()
with accelerator.main_process_first():
    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=batch_size,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
accelerator.wait_for_everyone()

train_dataset = processed_datasets["test"]

with accelerator.main_process_first():
    processed_datasets = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
eval_dataset = processed_datasets["test"] # 35k examples
test_dataset = processed_datasets["test"]

# processed_datasets = dataset.map(
#     preprocess_function,
#     batched=True,
#     batch_size=batch_size,
#     num_proc=1,
#     remove_columns=dataset["train"].column_names,
#     load_from_cache_file=False,
#     desc="Running tokenizer on dataset",
# )

# train_dataset = processed_datasets["test"]
# eval_dataset = processed_datasets["test"]

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

print("length of train dataloader: ", len(train_dataloader))
print("length of evaluation dataloader: ", len(eval_dataloader))


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

model, train_dataloader, eval_dataloader, test_dataloader, optimizer, lr_scheduler = accelerator.prepare(
    model, train_dataloader, eval_dataloader, eval_dataloader, optimizer, lr_scheduler
)
accelerator.print(model)

#######################
## Training and Evaluation
#######################

is_ds_zero_3 = False
if getattr(accelerator.state, "deepspeed_plugin", None):
    is_ds_zero_3 = accelerator.state.deepspeed_plugin.zero_stage == 3

import gc, psutil, threading

# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)

# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")

import time

for epoch in range(num_epochs):
    with TorchTracemalloc() as tracemalloc:
        model.train()
        total_loss = 0
        
        step_per_gb = gbs // batch_size
        num_global_step = len(train_dataloader) // step_per_gb
        start_time = time.time()

        for step, batch in enumerate(tqdm(train_dataloader)):
            global_step = step // step_per_gb
            micro_step = step % step_per_gb

            if global_step == 1 and micro_step == 0:
                step_time = time.time() - start_time
                print(f"Step time for {gbs} samples is {step_time} (sec)")
            if micro_step == 0:
                optimizer.zero_grad()
        
            outputs = model(**batch)
            loss = outputs.loss / step_per_gb
            accelerator.backward(loss)

            if micro_step == step_per_gb - 1:
                optimizer.step()
                lr_scheduler.step()

        # for global_step in range(num_global_step):
        #     if global_step == 1:
        #         step_time = time.time() - start_time
        #         print(f"Step time for {gbs} samples is {step_time} (sec)")
        #     optimizer.zero_grad()
        #     for micro_step in range(step_per_gb):
        #         batch = next(train_dataloader)
        #         outputs = model(**batch)
        #         loss = outputs.loss / step_per_gb
        #         accelerator.backward(loss)
        #     optimizer.step()
        #     lr_scheduler.step()

        # start_time = time.time()
        # for step, batch in enumerate(tqdm(train_dataloader)):
        #     if step == gbs // batch_size:
        #         step_time = time.time() - start_time
        #         print(f"Step time for {gbs} samples is {step_time} (sec)")
        #     outputs = model(**batch)
        #     loss = outputs.loss
        #     total_loss += loss.detach().float()
        #     accelerator.backward(loss)
        #     optimizer.step()
        #     lr_scheduler.step()
        #     optimizer.zero_grad()
    # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
    accelerator.print("GPU Memory before entering the train : {}".format(b2mb(tracemalloc.begin)))
    accelerator.print("GPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.used))
    accelerator.print("GPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.peaked))
    accelerator.print(
        "GPU Total Peak Memory consumed during the train (max): {}".format(
            tracemalloc.peaked + b2mb(tracemalloc.begin)
        )
    )

    accelerator.print("CPU Memory before entering the train : {}".format(b2mb(tracemalloc.cpu_begin)))
    accelerator.print("CPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.cpu_used))
    accelerator.print("CPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.cpu_peaked))
    accelerator.print(
        "CPU Total Peak Memory consumed during the train (max): {}".format(
            tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
        )
    )
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

    # model.eval()
    # eval_preds = []
    # with TorchTracemalloc() as tracemalloc:
    #     for _, batch in enumerate(tqdm(eval_dataloader)):
    #         batch = {k: v for k, v in batch.items() if k != "labels"}
    #         with torch.no_grad():
    #             outputs = accelerator.unwrap_model(model).generate(
    #                 **batch, synced_gpus=is_ds_zero_3, max_new_tokens=10
    #             )  # synced_gpus=True for DS-stage 3
    #         outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
    #         preds = accelerator.gather_for_metrics(outputs)
    #         preds = preds[:, max_length:].detach().cpu().numpy()
    #         eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

    # # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
    # accelerator.print("GPU Memory before entering the eval : {}".format(b2mb(tracemalloc.begin)))
    # accelerator.print("GPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.used))
    # accelerator.print("GPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.peaked))
    # accelerator.print(
    #     "GPU Total Peak Memory consumed during the eval (max): {}".format(
    #         tracemalloc.peaked + b2mb(tracemalloc.begin)
    #     )
    # )

    # accelerator.print("CPU Memory before entering the eval : {}".format(b2mb(tracemalloc.cpu_begin)))
    # accelerator.print("CPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.cpu_used))
    # accelerator.print("CPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.cpu_peaked))
    # accelerator.print(
    #     "CPU Total Peak Memory consumed during the eval (max): {}".format(
    #         tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
    #     )
    # )

    # correct = 0
    # total = 0
    # assert len(eval_preds) == len(
    #     dataset["train"][label_column]
    # ), f"{len(eval_preds)} != {len(dataset['train'][label_column])}"
    # for pred, true in zip(eval_preds, dataset["train"][label_column]):
    #     if pred.strip() == true.strip():
    #         correct += 1
    #     total += 1
    # accuracy = correct / total * 100
    # accelerator.print(f"{accuracy=}")
    # accelerator.print(f"{eval_preds[:10]=}")
    # accelerator.print(f"{dataset['train'][label_column][:10]=}")

#######################
## Export model
#######################

peft_model_id = f"{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"
model.save_pretrained(peft_model_id)

