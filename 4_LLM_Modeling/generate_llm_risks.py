#!/usr/bin/env python3

import argparse
import re
import time
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_message(note_text):
    system = """
You are a cardiologist.

The patient is from a heart failure clinic population.
Reduced LVEF, elevated BNP, and NYHA II–III symptoms are common.

Provide a thorough clinical risk assessment.

Discuss:
- Risk of sudden cardiac death (SCD)
- Risk of pump failure death (PFD)
- The patient's realistic likelihood of survival

Use specific clinical findings (LVEF, arrhythmias, BNP, NYHA class,
renal function, medications, ECG features, labs).

Avoid extreme language unless strongly supported by data.

Write approximately 250–450 words.
Do not use bullet points.
Do not use markdown.
Write in paragraph form.
"""

    prompt = f"Patient data:\n{note_text}"

    return [
        {"role": "system", "content": system.strip()},
        {"role": "user", "content": prompt.strip()},
    ]


ALLOWED_RISKS = {"low", "moderate", "high"}

def extract_risks(text):
    text_lower = text.lower()

    scd_match = re.search(
        r"(sudden cardiac death|scd)[^\.]{0,80}?(low|moderate|high)",
        text_lower
    )

    pfd_match = re.search(
        r"(pump failure death|pfd)[^\.]{0,80}?(low|moderate|high)",
        text_lower
    )

    scd_risk = scd_match.group(2) if scd_match else None
    pfd_risk = pfd_match.group(2) if pfd_match else None

    if scd_risk not in ALLOWED_RISKS:
        scd_risk = None
    if pfd_risk not in ALLOWED_RISKS:
        pfd_risk = None

    return scd_risk, pfd_risk


def generate_response(model, tokenizer, message, max_new_tokens=900):

    input_ids = tokenizer.apply_chat_template(
        message,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,                 # Greedy decoding (reproducible)
            repetition_penalty=1.1,          # Prevent loops
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output[0][input_ids.shape[-1]:]
    result = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(result)
    return result.strip()

def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    df = pd.read_csv(args.input_csv)

    if "Patient ID" not in df.columns:
        raise ValueError("CSV must contain 'Patient ID' column.")

    if "Prompts" not in df.columns:
        raise ValueError("CSV must contain 'Prompts' column.")

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing patients"):

        patient_id = row["Patient ID"]
        note_text = row["Prompts"]

        message = get_message(note_text)

        raw_output = None
        attempts = 0

        while attempts < args.max_retries:
            attempts += 1

            raw_output = generate_response(model, tokenizer, message)

            if raw_output:
                break

            tqdm.write(f"[WARN] Empty output for patient {patient_id}, retry {attempts}")
            time.sleep(0.2)

        if raw_output is None:
            tqdm.write(f"[ERROR] No output for patient {patient_id}")
            raw_output = ""

        scd_risk, pfd_risk = extract_risks(raw_output)


        results.append({
            "Patient ID": patient_id,
            "scd_risk": scd_risk,
            "pfd_risk": pfd_risk,
            "reasoning_text": raw_output,
            "raw_response": raw_output
        })

    output_df = pd.DataFrame(results)
    output_df.to_csv(args.output_csv, index=False)

    print(f"\nSaved results to {args.output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--max_retries", type=int, default=3)

    args = parser.parse_args()

    main(args)
