"""
замеры Phi-1.5 1.3b + LoRA на едином бенчмарке
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os
from peft import PeftModel

"""загрузка датасета"""

dataset_path = "data_results/step1_res/Phi-1.5_results_4k.csv"
df = pd.read_csv(dataset_path)

df.head()

df.info()

df

"""загрузка модели"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def loading_model():
    model_path = "./model_phi-1.5"  # локальный путь

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True
    )

    return tokenizer, model

def load_lora_adapter(base_model_name: str, adapter_path: str):

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    return model, tokenizer

"""тестирование модели на бенчмарках напрямую из общего датасета"""

import ast

def parse_options(options):
    if isinstance(options, str):
        return ast.literal_eval(options)
    return options

def build_prompt(row):

    prompt = ""
    if row["task_type"] == "summarization":
        if row["benchmark"] == "halueval_sum":
            document = row.get("context", "")

            prompt = f"""
You are a strict evaluator of hallucinations.

Document:
{row['question']}

Summary:
{row['correct_answer']}

Task:
Determine whether the summary contains hallucinated information.

Rules:
- Hallucination = any information NOT supported by the document
- Contradictions = hallucination
- Extra facts not in document = hallucination
- Missing information is OK

Answer:
YES → hallucination present
NO → no hallucination

Reply with only one word: YES or NO.
"""

            return prompt
        
        elif row["benchmark"] == "dailymail":
            document = row.get("context", "")

            prompt = f"""
Summarize the following news article.

Article:
{document}

Write a concise summary in 3-4 sentences.
Ensure the summary is factually accurate and based only on the article.
Do not add any external information or assumptions.
"""
            return prompt


        elif row["benchmark"] == "drop":

            context = row.get("context", "")

            prompt = f"""
Read the passage and answer the question.

Passage:
{context}

Question:
{row['question']}

Provide the final answer only (a number, date, or short phrase).
"""
            return prompt


    elif row["task_type"] == "qa":
        if row["benchmark"] == "truthfulqa" or row["benchmark"] == "bbh":

            prompt = f"""
Answer the question briefly.

Question:
{row['question']}
"""
            return prompt
            
        elif row["benchmark"] == "halueval_qa":
            
            prompt = f"""
You are a strict evaluator of hallucinations in question answering.

Context:
{row['context']}

Question:
{row['question']}

Answer:
{row['correct_answer']}

Task:
Determine whether the answer contains hallucinated information.

Rules:
- Hallucination = any information NOT supported by the context
- If the answer includes facts not in the context → hallucination
- If the answer contradicts the context → hallucination
- Missing details are OK

Decision:
YES → hallucination present
NO → fully supported by context

Reply with only one word: YES or NO.
"""

        return prompt



    elif row["benchmark"] == "mmlu":
        items = row['options']
        options = f"A. {row['options'][0]}\nB. {row['options'][1]}\nC. {row['options'][2]}\nD. {row['options'][3]}\n"
        prompt = f"""
Answer the question and choose one of the suggested answers. Answer format: "Answer: <letter>. Text of the selected option". Possible letters: A, B, C, D.

Question:
{row['question']}

Options:
{options}
"""
        return prompt


    elif row["benchmark"] == "fever":
        prompt = f"""
Determine whether the claim is supported by evidence.

Claim:
{row['question']}

Answer with:
SUPPORTS, REFUTES or NOT ENOUGH INFO
"""
        return prompt



    elif row["benchmark"] == "hover":
        prompt = f"""
Determine if the claim is true.

Claim:
{row['question']}

Answer with:
SUPPORTS or REFUTES
"""

        return prompt

    elif row["benchmark"] == "hellaswag":
        if isinstance(row['options'], str):
            items = ast.literal_eval(row['options'])
        else:
            items = row['options']


        options = "\n".join(f"{i}. {text}" for i, text in enumerate(items, 0))

        prompt = f"""
Choose the correct option.

Respond with ONLY ONE CHARACTER: 0, 1, 2, or 3.

DO NOT explain.

Question:
{row['question']}

Options:
{options}

Answer:
"""
        return prompt

def generate_answer(prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)

    output = model.generate(
        **inputs,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=100,
        do_sample=False,
        temperature=0.0
    )

    text = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    answer = text[len(prompt):].strip()

    return answer



"""добавление столбцов для ответов"""

# добавление столбцов для хранения результатов

models_list = ["Phi-1.5"]

attempt_list = [1]

model_path = "model_phi-1.5"
adapters = [
    "lora_models/Phi-1.5_att", 
    "lora_models/Phi-1.5_attmlp", 
    "lora_models/Phi-1.5_mlp"
]

import warnings
warnings.filterwarnings('ignore')

save_path = "data_results/step1_res/Phi-1.5_results_4k.csv"

for m in range(len(models_list)):
    for i in attempt_list:
        for adapter in adapters:
            counter = 0
            model, tokenizer = load_lora_adapter(model_path, adapter)
            model.eval()
            print("--------------------")
            print(model)
            print(tokenizer)
            
            if adapter == "lora_models/Phi-1.5_att":
                answer_col = f"{models_list[m]}_att_{i}"
            elif adapter == "lora_models/Phi-1.5_attmlp":
                answer_col = f"{models_list[m]}_attmlp_{i}"
            elif adapter == "lora_models/Phi-1.5_mlp":
                answer_col = f"{models_list[m]}_mlp_{i}"
            print(adapter)
            for idx, row in tqdm(df.iterrows(), total=len(df)):
                if pd.notna(df.at[idx, answer_col]):
                    counter += 1
                    continue

                prompt = build_prompt(row)
                answer = generate_answer(prompt)
                df.at[idx, answer_col] = answer
                
                counter += 1
                if counter % 1000 == 0:
                    df.to_csv(save_path, index=False)
                    print(f"обработано и сохранено {counter} задач")
            df.to_csv(save_path, index=False)
            print(f"обработано и сохранено {counter} задач")

print(df.isna().sum())
