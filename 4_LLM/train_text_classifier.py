#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# Model
# ============================================================

class BinaryClassifier(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


# ============================================================
# Load embeddings from folder
# ============================================================

def load_embeddings(embedding_dir, df, llm_model_name, lm_model_name):

    embeddings = []
    valid_rows = []

    print("\nLoading embeddings...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Patients"):
        pid = str(row["Patient ID"])
        fname = f"{pid}_{llm_model_name}_{lm_model_name}.npy"
        fpath = os.path.join(embedding_dir, fname)

        if not os.path.exists(fpath):
            print(f"Warning: Missing embedding for {pid}")
            continue

        emb = np.load(fpath)
        embeddings.append(emb)
        valid_rows.append(idx)

    df_valid = df.iloc[valid_rows].reset_index(drop=True)
    embeddings = np.vstack(embeddings)

    return embeddings, df_valid


# ============================================================
# Train one fold
# ============================================================

def train_one_fold(X_train, y_train, X_test, y_test, pos_weight, epochs=100):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BinaryClassifier(input_dim=X_train.shape[1]).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(device)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    model.train()
    epoch_bar = tqdm(range(epochs), desc="Training", leave=False)
    for _ in epoch_bar:
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        epoch_bar.set_postfix(loss=f"{loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits).cpu().numpy()

    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    return acc, auc, precision, recall, f1


# ============================================================
# Cross-validation
# ============================================================

def run_task(embeddings, df, positive_label, task_name):

    print(f"\n==============================")
    print(f"Task: Survivor vs {task_name}")
    print(f"==============================")

    mask = df["label"].isin([0, positive_label])
    df_task = df[mask].reset_index(drop=True)
    X = embeddings[mask]
    y = (df_task["label"] == positive_label).astype(int).values

    results = []

    folds = sorted(df_task["fold"].unique())

    for fold in tqdm(folds, desc=f"{task_name} Folds"):

        print(f"\nFold {fold}")

        train_idx = df_task["fold"] != fold
        test_idx = df_task["fold"] == fold

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        pos_weight = n_neg / n_pos

        acc, auc, precision, recall, f1 = train_one_fold(
            X_train, y_train,
            X_test, y_test,
            pos_weight
        )

        print(f"AUC: {auc:.4f} | F1: {f1:.4f}")

        results.append([acc, auc, precision, recall, f1])

    results = np.array(results)

    print(f"\n=== {task_name} Results (5-fold) ===")
    print(f"Accuracy:  {results[:,0].mean():.4f} ± {results[:,0].std():.4f}")
    print(f"AUC:       {results[:,1].mean():.4f} ± {results[:,1].std():.4f}")
    print(f"Precision: {results[:,2].mean():.4f} ± {results[:,2].std():.4f}")
    print(f"Recall:    {results[:,3].mean():.4f} ± {results[:,3].std():.4f}")
    print(f"F1:        {results[:,4].mean():.4f} ± {results[:,4].std():.4f}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dir", type=str, required=True)
    parser.add_argument("--fold_csv", type=str, required=True)
    parser.add_argument("--llm_model_name", type=str, required=True)
    parser.add_argument("--lm_model_name", type=str, required=True)
    args = parser.parse_args()

    print("Loading fold CSV...")
    df = pd.read_csv(args.fold_csv)

    embeddings, df = load_embeddings(
        args.embedding_dir,
        df,
        args.llm_model_name,
        args.lm_model_name
    )

    if len(embeddings) != len(df):
        raise ValueError("Mismatch between embeddings and dataframe.")

    run_task(embeddings, df, positive_label=3, task_name="SCD")
    run_task(embeddings, df, positive_label=6, task_name="PFD")


if __name__ == "__main__":
    main()
