#!/usr/bin/env python3

import os
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
import logging

# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


# ============================================================
# Models
# ============================================================

class LinearMultiHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.head_scd = nn.Linear(input_dim, 1)
        self.head_pfd = nn.Linear(input_dim, 1)

    def forward(self, x, return_embedding=False):
        if return_embedding:
            return x
        return self.head_scd(x).squeeze(-1), self.head_pfd(x).squeeze(-1)


class MLPMultiHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.2, layers=2):
        super().__init__()

        trunk = []
        in_dim = input_dim
        for _ in range(layers):
            trunk.append(nn.Linear(in_dim, hidden_dim))
            trunk.append(nn.ReLU())
            trunk.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.trunk = nn.Sequential(*trunk)

        self.head_scd = nn.Linear(hidden_dim, 1)
        self.head_pfd = nn.Linear(hidden_dim, 1)

    def forward(self, x, return_embedding=False):
        z = self.trunk(x)
        if return_embedding:
            return z
        return self.head_scd(z).squeeze(-1), self.head_pfd(z).squeeze(-1)


# ============================================================
# Evaluation
# ============================================================

def eval_head_np(y_true, y_prob):

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    mask = np.isfinite(y_prob)
    y_true = y_true[mask]
    y_prob = y_prob[mask]

    if len(y_true) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan

    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    return float(acc), float(prec), float(rec), float(f1), float(auc)


# ============================================================
# Load embeddings
# ============================================================

def load_embeddings(embedding_root, df, llm_model_name, lm_model_name):

    embeddings = []
    valid_rows = []

    embedding_dir = os.path.join(embedding_root, llm_model_name, lm_model_name)

    if not os.path.exists(embedding_dir):
        raise RuntimeError(f"Embedding directory does not exist:\n{embedding_dir}")

    print(f"\nLoading embeddings from: {embedding_dir}")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Patients"):
        pid = str(row["Patient ID"])
        fname = f"{pid}_{llm_model_name}_{lm_model_name}.npy"
        fpath = os.path.join(embedding_dir, fname)

        if not os.path.exists(fpath):
            continue

        emb = np.load(fpath)
        embeddings.append(emb)
        valid_rows.append(idx)

    if len(embeddings) == 0:
        raise RuntimeError(
            f"No embeddings found in {embedding_dir}. Check LLM/LM names."
        )

    df_valid = df.iloc[valid_rows].reset_index(drop=True)
    embeddings = np.vstack(embeddings)

    print(f"Loaded {len(embeddings)} embeddings.")
    return embeddings, df_valid


# ============================================================
# Save shared embeddings (ECG-style)
# ============================================================

@torch.no_grad()
def extract_and_save_embeddings_npz(
    model,
    X,
    y,
    pids,
    device,
    out_path,
):
    model.eval()

    X_t = torch.tensor(X, dtype=torch.float32).to(device)

    z = model(X_t, return_embedding=True)
    z = z.detach().cpu().numpy().astype(np.float32)

    y_scd = y[:, 0].astype(np.int64)
    y_pfd = y[:, 1].astype(np.int64)

    np.savez_compressed(
        out_path,
        pids=np.array(pids),
        z=z,
        y_scd=y_scd,
        y_pfd=y_pfd,
    )


# ============================================================
# Train one fold
# ============================================================

def train_one_fold(X_train, y_train, X_val, y_val, args):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.classifier == "linear":
        model = LinearMultiHead(X_train.shape[1]).to(device)
    else:
        model = MLPMultiHead(
            input_dim=X_train.shape[1],
            hidden_dim=args.mlp_hidden,
            dropout=args.mlp_dropout,
            layers=args.mlp_layers
        ).to(device)

    n_pos_scd = y_train[:, 0].sum()
    n_pos_pfd = y_train[:, 1].sum()
    n_neg = len(y_train) - ((y_train.sum(axis=1) > 0).sum())

    w_scd = torch.tensor([n_neg / max(n_pos_scd, 1)], device=device)
    w_pfd = torch.tensor([n_neg / max(n_pos_pfd, 1)], device=device)

    crit_scd = nn.BCEWithLogitsLoss(pos_weight=w_scd)
    crit_pfd = nn.BCEWithLogitsLoss(pos_weight=w_pfd)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    best_mean_auc = -np.inf
    best_model_state = None
    best_epoch = -1
    best_p_scd = None
    best_p_pfd = None

    for epoch in range(args.epochs):

        model.train()
        optimizer.zero_grad()

        z_scd, z_pfd = model(X_train_t)

        loss = (
            crit_scd(z_scd, y_train_t[:, 0]) +
            crit_pfd(z_pfd, y_train_t[:, 1])
        )

        loss.backward()
        optimizer.step()

        # ---- Validation ----
        model.eval()
        with torch.no_grad():
            z_scd, z_pfd = model(X_val_t)
            p_scd = torch.sigmoid(z_scd).cpu().numpy()
            p_pfd = torch.sigmoid(z_pfd).cpu().numpy()

        auc_s = roc_auc_score(y_val[:, 0], p_scd) if len(np.unique(y_val[:, 0])) > 1 else np.nan
        auc_p = roc_auc_score(y_val[:, 1], p_pfd) if len(np.unique(y_val[:, 1])) > 1 else np.nan
        mean_auc = np.nanmean([auc_s, auc_p])

        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            best_epoch = epoch + 1
            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_p_scd = p_scd.copy()
            best_p_pfd = p_pfd.copy()

    if best_model_state is None:
        raise RuntimeError("best_model_state was never set (no valid epoch).")
    model.load_state_dict(best_model_state)

    return (
        best_mean_auc,
        best_epoch,
        model,
        y_val[:, 0],
        best_p_scd,
        y_val[:, 1],
        best_p_pfd,
    )

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--embedding_dir", type=str, required=True)
    parser.add_argument("--fold_csv", type=str, required=True)
    parser.add_argument("--llm_model_name", type=str, required=True)
    parser.add_argument("--lm_model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="text_outputs")

    parser.add_argument("--text_embedding_dir", type=str, default=None)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")

    parser.add_argument("--classifier", type=str, choices=["linear", "mlp"], default="linear")
    parser.add_argument("--mlp_hidden", type=int, default=256)
    parser.add_argument("--mlp_dropout", type=float, default=0.2)
    parser.add_argument("--mlp_layers", type=int, default=2)

    args = parser.parse_args()

    set_seed(args.seed, deterministic=args.deterministic)

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.fold_csv)

    embeddings, df = load_embeddings(
        args.embedding_dir,
        df,
        args.llm_model_name,
        args.lm_model_name
    )

    folds = sorted(df["fold"].unique())

    for fold in folds:

        set_seed(args.seed + int(fold), deterministic=args.deterministic)

        train_mask = df["fold"] != fold
        val_mask = df["fold"] == fold

        X_train = embeddings[train_mask]
        X_val = embeddings[val_mask]

        y_train_raw = df[train_mask]["label"].values
        y_val_raw = df[val_mask]["label"].values

        y_train = np.zeros((len(y_train_raw), 2))
        y_val = np.zeros((len(y_val_raw), 2))

        y_train[:, 0] = (y_train_raw == 3).astype(int)
        y_train[:, 1] = (y_train_raw == 6).astype(int)

        y_val[:, 0] = (y_val_raw == 3).astype(int)
        y_val[:, 1] = (y_val_raw == 6).astype(int)

        (
            best_mean_auc,
            best_epoch,
            model,
            y_true_scd,
            y_prob_scd,
            y_true_pfd,
            y_prob_pfd,
        ) = train_one_fold(
            X_train, y_train,
            X_val, y_val,
            args
        )

        print(f"\nFold {fold} complete | Best mean AUC: {best_mean_auc:.4f}")

        fold_out = os.path.join(args.output_dir, f"val_fold_{fold}")
        os.makedirs(fold_out, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )
        
        # ---- Save best model weights ----
        torch.save(
            {"model_state_dict": model.state_dict()},
            os.path.join(fold_out, "best_model.pt"),
        )
        
        # ---- Save validation predictions ----
        np.savez_compressed(
            os.path.join(fold_out, "val_predictions.npz"),
            y_true_scd=y_true_scd.astype(np.int64),
            y_prob_scd=y_prob_scd.astype(np.float32),
            y_true_pfd=y_true_pfd.astype(np.int64),
            y_prob_pfd=y_prob_pfd.astype(np.float32),
        )
        
        print(f"Saved fold {fold} outputs to {fold_out}")

        # ------------------------------------------------------------
        # Recompute metrics from best predictions (consistent pattern)
        # ------------------------------------------------------------
        acc_s, prec_s, rec_s, f1_s, auc_s = eval_head_np(y_true_scd, y_prob_scd)
        acc_p, prec_p, rec_p, f1_p, auc_p = eval_head_np(y_true_pfd, y_prob_pfd)
        mean_auc = float(np.nanmean([auc_s, auc_p]))
        
        fold_summary = {
            "val_fold": int(fold),
            "best_epoch": int(best_epoch),
        
            # SCD metrics
            "scd_auc": auc_s,
            "scd_acc": acc_s,
            "scd_prec": prec_s,
            "scd_rec": rec_s,
            "scd_f1": f1_s,
        
            # PFD metrics
            "pfd_auc": auc_p,
            "pfd_acc": acc_p,
            "pfd_prec": prec_p,
            "pfd_rec": rec_p,
            "pfd_f1": f1_p,
        
            "mean_auc": mean_auc,
        }
        
        # include hyperparams + model names (very helpful later)
        for k, v in vars(args).items():
            fold_summary[f"param_{k}"] = v
        
        global_csv = os.path.join(args.output_dir, "cv_results_all_folds.csv")
        df_row = pd.DataFrame([fold_summary])
        
        if os.path.exists(global_csv):
            df_row.to_csv(global_csv, mode="a", header=False, index=False)
        else:
            df_row.to_csv(global_csv, index=False)
        
        print(
            f"Fold {fold} summary | mean_auc={mean_auc:.4f} | "
            f"scd_auc={auc_s:.4f} | pfd_auc={auc_p:.4f} | best_epoch={best_epoch}"
        )

        # ============================================================
        # Save Shared Text Embeddings
        # ============================================================

        if args.text_embedding_dir is not None:

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)

            fold_dir = os.path.join(
                args.text_embedding_dir,
                f"val_fold_{fold}"
            )
            os.makedirs(fold_dir, exist_ok=True)

            train_pids = df[train_mask]["Patient ID"].values
            val_pids = df[val_mask]["Patient ID"].values

            extract_and_save_embeddings_npz(
                model,
                X_train,
                y_train,
                train_pids,
                device,
                os.path.join(fold_dir, "train_embeddings.npz"),
            )

            extract_and_save_embeddings_npz(
                model,
                X_val,
                y_val,
                val_pids,
                device,
                os.path.join(fold_dir, "val_embeddings.npz"),
            )

            print(f"Saved text embeddings to: {fold_dir}")

    print("\nAll folds complete.")


if __name__ == "__main__":
    main()
