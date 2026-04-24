import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import json
from transformers import DataCollatorForSeq2Seq
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer
import shutil
import os
from peft import PeftModel
import matplotlib.pyplot as plt
import multiprocessing as mp
mp.set_start_method("spawn", force=True)
print(torch.cuda.is_available())

"""загрузка модели и применение LoRA"""

def loading_model_lora(model_path):

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = "{% for message in messages %}{{message['role']}}: {{message['content']}}\n{% endfor %}assistant: "

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        local_files_only=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model

def save_lora_adapter(model, output_path: str):
    model.save_pretrained(output_path)
    print(f"Adapter saved to {output_path}")

    adapter_size = sum(
        os.path.getsize(os.path.join(output_path, f))
        for f in os.listdir(output_path)
        if os.path.isfile(os.path.join(output_path, f))
    )
    print(f"Adapter size: {adapter_size / 1e6:.1f} MB")

def load_lora_adapter(base_model_name: str, adapter_path: str):

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    return model, tokenizer


def tokenize_chat(example, tokenizer, max_length=1024):
    messages = example["messages"]
    prompt_messages = messages[:-1]
    answer = messages[-1]["content"]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    full_text = prompt_text + answer

    tokenized_full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )

    tokenized_prompt = tokenizer(
        prompt_text,
        truncation=False,
        padding=False,
        add_special_tokens=False
    )

    input_ids = tokenized_full["input_ids"]
    labels = input_ids.copy()
    prompt_len = min(len(tokenized_prompt["input_ids"]), len(input_ids))
    labels[:prompt_len] = [-100] * prompt_len
    labels = [
        (-100 if token_id == tokenizer.pad_token_id else label)
        for token_id, label in zip(input_ids, labels)
    ]

    return {
        "input_ids": input_ids,
        "attention_mask": tokenized_full["attention_mask"],
        "labels": labels,
    }

def apply_lora(model, lora_target_modules):
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules= lora_target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model

def get_training_args(model_name, modules):
    return TrainingArguments(
        output_dir= f"./lora_models/{model_name}_{modules}",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate= 3e-4,
        num_train_epochs=2,
        logging_steps=10,     # 100
        logging_dir= f"./lora_models/{model_name}_{modules}logs",
        disable_tqdm=False,
        save_steps=500,
        fp16=True,
        bf16=False,
        lr_scheduler_type="cosine",
        warmup_steps=1,
        label_names=["labels"],
        gradient_checkpointing=False,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0
    )

def prepare_dataset(dataset_path, tokenizer):
    dataset = load_dataset("json", data_files=dataset_path)["train"]

    dataset = dataset.map(
        lambda x: tokenize_chat(x, tokenizer),
        remove_columns=dataset.column_names
    )

    return dataset

def train(model, tokenizer, dataset, model_name, modules):
    training_args = get_training_args(model_name, modules)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    trainer.train()

    logs = trainer.state.log_history
    plot_loss(model_name, modules, logs)

    model.save_pretrained(f"./lora_models/{model_name}_{modules}")
    tokenizer.save_pretrained(f"./lora_models/{model_name}_{modules}")


def plot_loss(model_name, modules, logs):
    train_loss = []
    steps = []

    for log in logs:
        if "loss" in log:
            train_loss.append(log["loss"])
            steps.append(log["step"])

    plt.figure()
    plt.plot(steps, train_loss)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid()
    plt.savefig(f"./lora_models/{model_name}_{modules}_loss_plot.png")
    plt.show()
    






"""загрузка датасета для дообучения и параметры"""

models_list = ["Pythia_410m","Qwen2.5_0.5b"]

model_path = ["model_pythia_410m","model_qwen2_5_0_5b"]

lora_dataset_path = "lora_dataset.json"

modules = ["att", "mlp", "attmlp"]

"""дообучение модели"""

for j in range (len(models_list)):
    if models_list[j] == "GPT_Neo_125m":
        lora_target_modules = [
            ["q_proj", "k_proj", "v_proj", "out_proj"],  # attention
            ["c_fc", "c_proj"],                         # mlp
            ["q_proj", "k_proj", "v_proj", "out_proj", "c_fc", "c_proj"]  # attention + mlp
        ]
    elif models_list[j] == "Phi-1.5":
        lora_target_modules = [
            ["q_proj", "k_proj", "v_proj", "dense"],
            ["fc1", "fc2"],
            ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        ]
    elif models_list[j] == "Pythia_410m":
        lora_target_modules = [
            ["query_key_value", "dense"],
            ["dense_h_to_4h", "dense_4h_to_h"],
            ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        ]
    elif models_list[j] == "Qwen2.5_0.5b":
        lora_target_modules = [
            ["q_proj", "k_proj", "v_proj", "o_proj"],
            ["gate_proj", "up_proj", "down_proj"],
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        ]
    elif models_list[j] == "SmolLM2_135m":
        lora_target_modules = [
            ["q_proj", "k_proj",  "v_proj", "o_proj"],
            ["gate_proj", "up_proj", "down_proj"],
            ["q_proj", "k_proj",  "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        ]
    elif models_list[j] == "TinyLlama":
        lora_target_modules = [
            ["q_proj", "k_proj",  "v_proj", "o_proj"],
            ["gate_proj", "up_proj", "down_proj"],
            ["q_proj", "k_proj",  "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        ]



    for i in range(len(lora_target_modules)):
        model_name = models_list[j]
        print("--------------------")
        print(modules[i])
        tokenizer, model = loading_model_lora(model_path[j])
        print(model)

        model = apply_lora(model, lora_target_modules[i])
        dataset = prepare_dataset(lora_dataset_path, tokenizer)
        print(next(model.parameters()).device)

        train(model, tokenizer, dataset, model_name, modules[i])
        print(model)

        """сохранение модели"""

        shutil.make_archive(
            f"./lora_models/{model_name}_{modules[i]}",
            'zip',
            f"./lora_models/{model_name}_{modules[i]}"
        )
        save_lora_adapter(model, f"./lora_models/{model_name}_{modules[i]}/adapter")

