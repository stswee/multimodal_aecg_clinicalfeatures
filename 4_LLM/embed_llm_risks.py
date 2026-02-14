#!/usr/bin/env python3
"""
Generate patient-level text embeddings from LLM explanations
using BioBERT and ClinicalBERT with CLS pooling.

Input:
- CSV file with one patient per row
- Required columns:
    - patient_id column (e.g., "Patient ID")
    - explanation column (e.g., "llm_explanation")

Output:
- One .npy embedding per patient per LM:
  lm_embeddings/{LLM_MODEL}/{LM_MODEL}/{patient_id}_{LLM_MODEL}_{LM_MODEL}.npy
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# -----------------------------
# Argument parsing
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Embed LLM explanations with BioBERT / ClinicalBERT (CLS pooling)"
    )

    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to CSV file")

    parser.add_argument("--patient_id_col", type=str, required=True,
                        help="Column name for patient ID")

    parser.add_argument("--text_col", type=str, required=True,
                        help="Column name for explanation text")

    parser.add_argument("--output_root", type=str, default="lm_embeddings",
                        help="Root directory for embeddings")

    parser.add_argument("--llm_model_name", type=str, default="LLaMA8B",
                        help="Name of upstream LLM (used in folder + filename)")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()

    # LM checkpoints
    LM_MODELS = {
        "BioBERT": "dmis-lab/biobert-base-cased-v1.1",
        "ClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT"
    }

    # -----------------------------
    # Load CSV
    # -----------------------------
    df = pd.read_csv(args.csv_path)

    if args.patient_id_col not in df.columns:
        raise ValueError(f"Column '{args.patient_id_col}' not found in CSV.")

    if args.text_col not in df.columns:
        raise ValueError(f"Column '{args.text_col}' not found in CSV.")

    patient_ids = df[args.patient_id_col].astype(str).tolist()
    texts = df[args.text_col].fillna("").astype(str).tolist()

    print(f"Loaded {len(texts)} patient explanations")
    print(f"Using device: {args.device}")

    # -----------------------------
    # Loop over language models
    # -----------------------------
    for lm_name, lm_ckpt in LM_MODELS.items():

        print(f"\nEmbedding with {lm_name} ({lm_ckpt})")

        out_dir = os.path.join(
            args.output_root,
            args.llm_model_name,
            lm_name
        )
        os.makedirs(out_dir, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(lm_ckpt)
        model = AutoModel.from_pretrained(lm_ckpt)

        model.to(args.device)
        model.eval()

        # Batch processing
        for i in tqdm(range(0, len(texts), args.batch_size)):
            batch_texts = texts[i:i + args.batch_size]
            batch_ids = patient_ids[i:i + args.batch_size]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt"
            ).to(args.device)

            with torch.no_grad():
                outputs = model(**inputs)

                # CLS pooling
                embeddings = outputs.last_hidden_state[:, 0, :]

            embeddings = embeddings.cpu().numpy()

            for emb, pid in zip(embeddings, batch_ids):
                fname = f"{pid}_{args.llm_model_name}_{lm_name}.npy"
                fpath = os.path.join(out_dir, fname)
                np.save(fpath, emb)

        print(f"Saved embeddings to {out_dir}")

    print("\nAll embeddings complete.")


if __name__ == "__main__":
    main()
