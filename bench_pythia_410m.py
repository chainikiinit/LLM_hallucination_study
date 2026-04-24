"""
замеры Pythia-410m на едином бенчмарке
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import json

"""загрузка датасета"""

dataset_path = "./sampled_dataset_3.8k.csv"
df = pd.read_csv(dataset_path)

df.head()

df.info()

df

"""загрузка модели"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def loading_model():
    model_path = "./model_pythia_410m"  # локальный путь

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

models_list = ["Pythia_410m"]

def add_model_answer_and_result_columns(df, models_list, attempt_list):
    for m in models_list:
        for attempt in attempt_list:
            answer_col = f"{m}_{attempt}"
            result_col = f"{m}_{attempt}_result"
            answer_lora_col = f"{m}_LoRA_{attempt}"
            result_lora_col = f"{m}_LoRA_{attempt}_result"
            df[answer_col] = None           # для ответа модели
            df[result_col] = None          # для хранения результата (true/false)
            df[answer_lora_col] = None
            df[result_lora_col] = None
    return df

attempt_list = [1]
test_df = add_model_answer_and_result_columns(df, models_list, attempt_list)

test_df.columns.tolist()

tokenizer, model = loading_model()
model.eval()

model

import warnings
warnings.filterwarnings('ignore')

counter = 0
save_path = "data_results/sampled_dataset_3.8k_pythia.csv"

for m in range(len(models_list)):
    for i in attempt_list:
        answer_col = f"{models_list[m]}_{i}"
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
            if pd.notna(test_df.at[idx, answer_col]):
                continue

            prompt = build_prompt(row)
            answer = generate_answer(prompt)
            test_df.at[idx, answer_col] = answer

            counter += 1
            if counter % 1000 == 0:
                test_df.to_csv(save_path, index=False)
                print(f"обработано и сохранено {counter} задач")

print(test_df.isna().sum())

test_df.to_csv(
    "data_results/pythia_410m_results.csv",
    index=False
)