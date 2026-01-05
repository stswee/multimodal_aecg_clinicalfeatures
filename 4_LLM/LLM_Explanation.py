#!/usr/bin/env python3
"""
Generate joint SCD / PFD clinical explanations for all patients
using an instruction-tuned LLaMA-family model.

Outputs one explanation per patient (shared representation),
suitable for downstream embedding + MIL multi-head classification.
"""

# Import packages
import os
import random
import json
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Set seeds
random.seed(0)
torch.manual_seed(0)

# Load prompts
def load_prompt(prompt_dir: str):
    """
    Load system and user prompt templates from folder.
    Expects:
      - system_prompt.txt
      - user_prompt.txt
    """
    system_path = os.path.join(prompt_dir, "system_prompt.txt")
    user_path = os.path.join(prompt_dir, "user_prompt.txt")

    if not os.path.exists(system_path):
        raise FileNotFoundError(f"Missing system_prompt.txt in {prompt_dir}")
    if not os.path.exists(user_path):
        raise FileNotFoundError(f"Missing user_prompt.txt in {prompt_dir}")

    with open(system_path, "r") as f:
        system_prompt = f.read().strip()

    with open(user_path, "r") as f:
        user_prompt_template = f.read().strip()

    return system_prompt, user_prompt_template


def build_user_prompt(template: str, clinical_note: str) -> str:
    return template.replace("{{CLINICAL_NOTE}}", clinical_note)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True, type=str)
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--prompt_dir", required=True, type=str)
    parser.add_argument("--max_new_tokens", default=300, type=int)
    parser.add_argument("--id_column", default="Patient ID", type=str)
    parser.add_argument("--text_column", default="Prompts", type=str)
    parser.add_argument("--output_jsonl", default="llm_explanations.jsonl", type=str)

    args = parser.parse_args()

    # ---- HuggingFace login ----
    with open("../../huggingface_token.txt", "r") as f:
        token = f.read().strip()
    login(token)

    # ---- Load data ----
    df = pd.read_csv(
        args.csv_path,
        dtype={args.id_column: str}
    ).reset_index(drop=True)

    # 4-digit IDs (for MUSIC)
    df[args.id_column] = df[args.id_column].str.strip().str.zfill(4)

    # ---- Load prompts ----
    system_prompt, user_prompt_template = load_prompt(args.prompt_dir)

    # ---- Load model ----
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        token=token,
        use_fast=False
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        token=token,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()

    # ---- Resume-safe output ----
    processed_ids = set()
    if os.path.exists(args.output_jsonl):
        with open(args.output_jsonl, "r") as f:
            for line in f:
                processed_ids.add(json.loads(line)[args.id_column])

    out_f = open(args.output_jsonl, "a")

    # ---- Iterate patients ----
    for _, row in tqdm(df.iterrows(), total=len(df)):

        subject_id = row[args.id_column]
        if subject_id in processed_ids:
            continue

        clinical_note = row[args.text_column]
        if not isinstance(clinical_note, str) or clinical_note.strip() == "":
            print(f"Skipping patient {subject_id}: missing clinical note")
            continue

        user_prompt = build_user_prompt(user_prompt_template, clinical_note)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 1. Format chat as text
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False
        )
        
        # 2. Tokenize explicitly (this always returns input_ids + attention_mask)
        encodings = tokenizer(
            chat_text,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
            padding=False
        )
        
        input_ids = encodings["input_ids"].to(model.device)
        attention_mask = encodings["attention_mask"].to(model.device)


        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )


        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_text = decoded.split("assistant")[-1].strip()

        record = {
            args.id_column: subject_id,
            "llm_explanation": assistant_text,
        }

        out_f.write(json.dumps(record) + "\n")
        out_f.flush()

    out_f.close()


if __name__ == "__main__":
    main()