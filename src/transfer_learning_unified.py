#!/usr/bin/env python3
"""
Usage examples:
python transfer_learning_unified.py train \
  --pipeline llm --use_sum --sum_model medgemma-27b-text-it \
  --embedding_model Qwen3-Embedding-8B \
  --model_type mlp --task mort

python transfer_learning_unified.py test \
  --pipeline baseline --reps mean,right,interp \
  --model_type mlp --task forecast --finetune --finetune_size 16

python transfer_learning_unified.py train \
  --pipeline embedding --embedding_model TFM \
  --model_type lstm --task forecast

python transfer_learning_unified.py test \
  --pipeline embedding --embedding_model TSDE \
  --model_type lstm --task lab --use_temporal

Note that finetune is done on test
"""

import os, argparse
import torch
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

# ----------------- Imports from your project -----------------
from data_loaders.datasets.ppicu import PpicuLLMDataset, PpicuDataset
from data_loaders.datasets.hirid import HiridLLMDataset, HiridDataset
from data_loaders.datasets.mimic import MimicLLMDataset, MimicDataset
from exp.task_trainer import Trainer
from train_experiments import initialize_model

# -------------------------------------------------------------

# ---------- Dataset shapes (tabular baselines / labels) ----------
DATASET_PARAMS = {
    "mimic": {"D": 60, "T": 48, "T2": 24, "Lab": 21},
    "hirid": {"D": 64, "T": 48, "T2": 24, "Lab": 24},
    "ppicu": {"D": 75, "T": 48, "T2": 24, "Lab": 25},
}

# ---------- TSDE embedding dims per dataset (for embedding-only pipeline) ----------
TSDE_EMBED_D = {
    "hirid": 2112,
    "mimic": 1980,
    "ppicu": 2475,
}


# ---------- Utility: dims ----------
def get_tabular_input_dim(dataset: str) -> int:
    params = DATASET_PARAMS.get(dataset)
    if params is None:
        raise ValueError(f"Unknown dataset: {dataset}")
    return params["D"]


def get_embedding_input_dim(embedding_model: str, dataset: str) -> int:
    """
    Embed dim used by the decoder for the *embedding-only* pipeline.
    - TFM: 1280
    - TSDE: dataset-specific (TSDE_EMBED_D)
    - Otherwise: fallback to TSDE dims if known, else 4096 (safe high)
    """
    if embedding_model == "TFM":
        return 1280
    if embedding_model == "TSDE":
        return TSDE_EMBED_D[dataset]
    return 4096


# ---------- Pad/Truncate helpers ----------
def _pad_truncate_to(
    x: torch.Tensor, target_shape: torch.Size, pad_value: float = 0.0
) -> torch.Tensor:
    assert x.dim() == len(
        target_shape
    ), f"x.dim={x.dim()} vs target_shape={target_shape}"
    out = x
    slicers = [slice(0, min(out.size(d), target_shape[d])) for d in range(out.dim())]
    out = out[tuple(slicers)]
    pads = []
    for d in reversed(range(out.dim())):
        add = max(target_shape[d] - out.size(d), 0)
        pads.extend([0, add])
    if any(pads):
        out = F.pad(out, pads, mode="constant", value=pad_value)
    slicers = [slice(0, target_shape[d]) for d in range(out.dim())]
    return out[tuple(slicers)]


def _overlap_mask_like(targets: torch.Tensor, orig_shape: torch.Size) -> torch.Tensor:
    mask = torch.zeros_like(targets, dtype=targets.dtype, device=targets.device)
    if targets.dim() == 2:
        T = min(targets.size(1), orig_shape[1])
        mask[:, :T] = 1
    elif targets.dim() == 3:
        T = min(targets.size(1), orig_shape[1])
        D = min(targets.size(2), orig_shape[2])
        mask[:, :T, :D] = 1
    else:
        slicers = [slice(None)]
        for d in range(1, targets.dim()):
            slicers.append(slice(0, min(targets.size(d), orig_shape[d])))
        mask[tuple(slicers)] = 1
    return mask


# ---------- Multilabel thresholding ----------
def best_per_label_thresholds(
    val_logits: torch.Tensor, val_targets: torch.Tensor, n_steps: int = 1001
) -> torch.Tensor:
    probs = torch.sigmoid(val_logits)
    _, C = probs.shape
    ts = torch.linspace(0, 1, steps=n_steps, device=probs.device)
    best = torch.full((C,), 0.5, device=probs.device)
    for c in range(C):
        p = probs[:, c]
        y = val_targets[:, c].bool()
        best_f1, best_t = -1.0, 0.5
        for t in ts:
            preds = p >= t
            tp = (preds & y).sum().float()
            fp = (preds & ~y).sum().float()
            fn = (~preds & y).sum().float()
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            f1 = 2 * prec * rec / (prec + rec + 1e-9)
            if f1 > best_f1:
                best_f1, best_t = float(f1), float(t)
        best[c] = best_t
    return best


# ---------- Adjust feature dims in a batch ----------
def adjust_input(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    cur = x.shape[-1]
    if cur == target_dim:
        return x
    if cur > target_dim:
        return x[..., :target_dim]
    pad = torch.zeros(*x.shape[:-1], target_dim - cur, device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=-1)


def make_collate_fn(target_dim: int):
    def _collate(batch):
        coll = torch.utils.data.dataloader.default_collate(batch)
        coll[0] = adjust_input(coll[0], target_dim)
        return coll

    return _collate


# ---------- DataLoader factory ----------
def build_dataset(
    rep_datapath: str,
    data_path: str,
    dst: str,
    split: str,
    args,
    representation: str,
    train_percentile_mask: str = "all",
):
    split_eff = split if split != "finetune" else "train"

    if getattr(args, "pipeline", None) == "embedding":
        representation = args.embedding_model

    if args.embedding_model:
        # LLM embeddings / TFM / TSDE path
        if dst == "ppicu":
            return PpicuLLMDataset(
                rep_datapath,
                data_path,
                split_eff,
                args.embedding_model,
                representation,
                train_percentile_mask,
                args.seed,
            )
        if dst == "hirid":
            return HiridLLMDataset(
                rep_datapath,
                data_path,
                split_eff,
                args.embedding_model,
                representation,
                train_percentile_mask,
                args.seed,
            )
        if dst == "mimic":
            return MimicLLMDataset(
                rep_datapath,
                data_path,
                split_eff,
                args.embedding_model,
                representation,
                train_percentile_mask,
                args.seed,
            )
        raise NotImplementedError(dst)
    else:
        # Baselines / tabular
        if dst == "hirid":
            return HiridDataset(
                rep_datapath, data_path, split_eff, train_percentile_mask, args.seed
            )
        if dst == "mimic":
            return MimicDataset(
                rep_datapath, data_path, split_eff, train_percentile_mask, args.seed
            )
        if dst == "ppicu":
            return PpicuDataset(
                rep_datapath, data_path, split_eff, train_percentile_mask, args.seed
            )
        raise NotImplementedError(dst)


def get_loader(
    rep_datapath: str,
    data_path: str,
    dst: str,
    split: str,
    args,
    representation: str,
    train_percentile_mask: str = "all",
) -> DataLoader:
    ds = build_dataset(
        rep_datapath, data_path, dst, split, args, representation, train_percentile_mask
    )
    if args.finetune and split == "finetune":
        k = min(args.finetune_size, len(ds))
        sampler = SubsetRandomSampler(range(k))
        return DataLoader(
            ds,
            batch_size=min(k, args.batch_size),
            sampler=sampler,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
    else:
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
            pin_memory=True,
        )


# ---------- Representation path builders ----------
def rep_path_for_embedding(rep_dir: str, dataset: str, embedding_model: str) -> str:
    return os.path.join(rep_dir, f"{dataset}/{dataset}_{embedding_model}.npz")


def rep_path_for_llm_rep(
    rep_dir: str, dataset: str, emb_model: str, rep: str, use_sum: bool, sum_model: str
) -> str:
    if not use_sum:
        return os.path.join(rep_dir, f"{dataset}/{dataset}_all_{emb_model}_{rep}.npz")
    return os.path.join(
        rep_dir, f"{dataset}/{dataset}_all_sum_{sum_model}_{emb_model}_{rep}.npz"
    )


def rep_path_for_baseline(data_dir: str, dataset: str, rep: str) -> str:
    return os.path.join(data_dir, f"{dataset}/{dataset}_{rep}.npz")


def labels_npz_path(data_dir: str, dataset: str) -> str:
    return os.path.join(data_dir, f"{dataset}/{dataset}_mean_with_lab.npz")


# ---------- Evaluation ----------
def evaluate(
    model: torch.nn.Module,
    decoder: torch.nn.Module,
    loader: DataLoader,
    trainer: Trainer,
    task: str,
    use_temporal: bool,
    ml_thresholding: str = "per_label",
) -> dict:

    device = trainer.device
    model.to(device).eval()
    # Metrics
    if task in ["forecast", "los", "age"]:
        mae = torchmetrics.MeanAbsoluteError().to(device)
        mse = torchmetrics.MeanSquaredError().to(device)
        tot_abs = tot_sq = tot_cnt = 0.0
    elif task in ["mort", "gender"]:
        acc = torchmetrics.Accuracy(task="binary", threshold=0.5).to(device)
        auroc = torchmetrics.AUROC(task="binary").to(device)
        auprc = torchmetrics.AveragePrecision(task="binary").to(device)
    elif task in ["lab", "feature"]:
        ml_num = None
        ml_f1 = ml_prec = ml_rec = ml_auroc = ml_auprc = None
    else:
        raise ValueError(f"Unsupported task: {task}")

    S = getattr(trainer, "S", None)

    with torch.no_grad():
        for batch in loader:
            emb = batch[0].to(device).float()

            if decoder is None:
                dec = emb
            else:
                flat = emb.view(emb.size(0), -1)
                dec_vec = decoder(flat)
                if use_temporal and S is None:
                    dec = dec_vec
                else:
                    if S is None:
                        S = trainer.S
                    dec = dec_vec.unsqueeze(1).expand(-1, S, -1)

            outputs = model(dec).contiguous()
            targets = trainer._get_targets(batch).to(device).contiguous()

            # scalar-regression shape cleanup
            if task in ["los", "age"]:
                if outputs.ndim == 2 and outputs.size(-1) == 1:
                    outputs = outputs.squeeze(-1)
                if targets.ndim == 2 and targets.size(-1) == 1:
                    targets = targets.squeeze(-1)
                outputs = outputs.float()
                targets = targets.float()

            if task in ["los", "age"]:
                mae.update(outputs, targets)
                mse.update(outputs, targets)

            if task == "forecast":
                outputs = outputs.float()
                targets = targets.float()
                orig_shape = outputs.shape
                aligned = _pad_truncate_to(outputs, targets.shape, 0.0)

                base_mask = batch[2].to(device).to(dtype=targets.dtype).contiguous()
                # Safe overlap mask (strict as in embedding script)
                overlap = _overlap_mask_like(targets, orig_shape).to(targets.dtype)
                final_mask = base_mask * overlap

                valid = final_mask > 0.5
                if valid.any():
                    mae.update(aligned[valid], targets[valid])
                    mse.update(aligned[valid], targets[valid])

                diff = aligned - targets
                tot_abs += torch.abs(diff).mul(final_mask).sum().item()
                tot_sq += (diff * diff).mul(final_mask).sum().item()
                tot_cnt += base_mask.sum().item()
                continue

            if task in ["mort", "gender"]:
                logits = outputs
                if logits.ndim == 2 and logits.size(-1) == 1:
                    logits = logits.squeeze(-1)
                if targets.ndim == 2 and targets.size(-1) == 1:
                    targets = targets.squeeze(-1)
                y = targets.int()
                probs = torch.sigmoid(logits)
                acc.update(probs, y)
                auroc.update(probs, y)
                auprc.update(probs, y)

            if task in ["lab", "feature"]:
                logits = outputs.float()
                y_full = targets.int()
                Lm, Lt = logits.size(-1), y_full.size(-1)
                L = min(Lm, Lt)
                if L == 0:
                    continue
                logits = logits[:, :L]
                y = y_full[:, :L]

                if ml_num is None:
                    ml_num = L
                    ml_f1 = torchmetrics.F1Score(
                        task="multilabel", num_labels=L, average="micro", threshold=0.5
                    ).to(device)
                    ml_prec = torchmetrics.Precision(
                        task="multilabel", num_labels=L, average="micro", threshold=0.5
                    ).to(device)
                    ml_rec = torchmetrics.Recall(
                        task="multilabel", num_labels=L, average="micro", threshold=0.5
                    ).to(device)
                    ml_auroc = torchmetrics.AUROC(
                        task="multilabel", num_labels=L, average="micro"
                    ).to(device)
                    ml_auprc = torchmetrics.AveragePrecision(
                        task="multilabel", num_labels=L, average="micro"
                    ).to(device)

                probs = torch.sigmoid(logits)
                if ml_thresholding == "per_label":
                    thr = best_per_label_thresholds(logits, y)
                    preds = (probs >= thr).to(torch.float32)
                else:
                    preds = probs  # use raw probs for PR/ROC; F1/Prec/Rec will use threshold=0.5 set in metrics

                ml_f1.update(preds, y)
                ml_prec.update(preds, y)
                ml_rec.update(preds, y)
                ml_auroc.update(probs, y)  # AUROC with probabilities
                ml_auprc.update(probs, y)
                continue

    # collect
    out = {}
    if task in ["forecast", "los", "age"]:
        out["mae"] = round(mae.compute().item(), 5)
        out["mse"] = round(mse.compute().item(), 5)
        if task == "forecast":
            if tot_cnt > 0:
                out["masked_mae"] = round(tot_abs / tot_cnt, 5)
                out["masked_mse"] = round(tot_sq / tot_cnt, 5)
            else:
                out["masked_mae"] = 0.0
                out["masked_mse"] = 0.0
    elif task in ["mort", "gender"]:
        out["accuracy"] = round(acc.compute().item(), 5)
        out["auroc"] = round(auroc.compute().item(), 5)
        out["auprc"] = round(auprc.compute().item(), 5)
    else:
        out["f1_micro"] = round(ml_f1.compute().item(), 5)
        out["precision_micro"] = round(ml_prec.compute().item(), 5)
        out["recall_micro"] = round(ml_rec.compute().item(), 5)
        out["auroc_micro"] = round(ml_auroc.compute().item(), 5)
        out["auprc_micro"] = round(ml_auprc.compute().item(), 5)
    return out


# ---------- Train/Test/Finetune loops ----------
def train_all(datasets, reps, args, task):
    """
    reps:
      - If args.pipeline == 'embedding': reps is a single-element list [args.embedding_model]
      - If 'llm': reps like ['zero_shot','ICD','Trend','CoT']
      - If 'baseline': reps like ['mean','right','interp']
    """
    for rep in reps:
        for train_ds in datasets:
            if args.pipeline == "baseline":
                config = {
                    "dataset": train_ds,
                    "task": task,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "patience": args.patience,
                    "train_mask": "all",
                    "model_type": args.model_type,
                    "seed": args.seed,
                    "representation": rep,
                }
                rep_path = rep_path_for_baseline(args.data_dir, train_ds, rep)
                rep_str_for_model = rep
                embed_dim = get_tabular_input_dim(train_ds)

            elif args.pipeline == "llm":
                config = {
                    "dataset": train_ds,
                    "task": task,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "patience": args.patience,
                    "train_mask": "all",
                    "model_type": args.model_type,
                    "seed": args.seed,
                    "representation": (
                        f"all_{rep}" if not args.use_sum else f"all_sum_{rep}"
                    ),
                    "embed_dim": 4096,  # keep as before
                    "embedding_model": args.embedding_model or "Qwen3-Embedding-8B",
                    "summarization_model": args.sum_model,
                }
                rep_path = rep_path_for_llm_rep(
                    args.rep_dir,
                    train_ds,
                    config["embedding_model"],
                    rep,
                    args.use_sum,
                    args.sum_model,
                )
                rep_str_for_model = config["representation"]
                embed_dim = config["embed_dim"]

            else:  # embedding pipeline
                config = {
                    "dataset": train_ds,
                    "task": task,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "patience": args.patience,
                    "train_mask": "all",
                    "model_type": args.model_type,
                    "seed": args.seed,
                    "representation": args.embedding_model,
                    "embed_dim": get_embedding_input_dim(
                        args.embedding_model, train_ds
                    ),
                    "embedding_model": args.embedding_model,
                }
                rep_path = rep_path_for_embedding(
                    args.rep_dir, train_ds, args.embedding_model
                )
                rep_str_for_model = args.embedding_model
                embed_dim = config["embed_dim"]

            print(f"\n=== TRAINING rep={rep_str_for_model} on {train_ds} ===")
            model = initialize_model(
                args.model_type, task, train_ds, rep_str_for_model, args.embedding_model
            )
            trainer = Trainer(args, model, config)

            data_npz = labels_npz_path(args.data_dir, train_ds)
            train_loader = get_loader(
                rep_path, data_npz, train_ds, "train", args, "all"
            )
            val_loader = get_loader(rep_path, data_npz, train_ds, "val", args, "all")
            trainer.train(train_loader, val_loader)


def test_all(datasets, reps, args, task):
    for rep in reps:
        for train_ds in datasets:
            # Build config + model per rep type
            if args.pipeline == "baseline":
                config = {
                    "dataset": train_ds,
                    "task": task,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "patience": args.patience,
                    "train_mask": "all",
                    "model_type": args.model_type,
                    "seed": args.seed,
                    "representation": rep,
                }
                rep_str_for_model = rep
            elif args.pipeline == "llm":
                config = {
                    "dataset": train_ds,
                    "task": task,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "patience": args.patience,
                    "train_mask": "all",
                    "model_type": args.model_type,
                    "seed": args.seed,
                    "representation": (
                        f"all_{rep}" if not args.use_sum else f"all_sum_{rep}"
                    ),
                    "embed_dim": 4096,
                    "embedding_model": args.embedding_model or "Qwen3-Embedding-8B",
                    "summarization_model": args.sum_model,
                }
                rep_str_for_model = config["representation"]
            else:
                config = {
                    "dataset": train_ds,
                    "task": task,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "patience": args.patience,
                    "train_mask": "all",
                    "model_type": args.model_type,
                    "seed": args.seed,
                    "representation": args.embedding_model,
                    "embed_dim": get_embedding_input_dim(
                        args.embedding_model, train_ds
                    ),
                    "embedding_model": args.embedding_model,
                }
                rep_str_for_model = args.embedding_model

            model = initialize_model(
                args.model_type, task, train_ds, rep_str_for_model, args.embedding_model
            )
            trainer = Trainer(args, model, config)
            decoder = getattr(trainer, "decoder", None)

            ckpt_name = f"best_{trainer.experiment_string}.pt"
            ckpt_path = os.path.join(trainer.save_dir, ckpt_name)
            if not os.path.exists(ckpt_path):
                print(
                    f"[SKIP] No checkpoint for train_ds={train_ds}, rep={rep_str_for_model}: {ckpt_path}"
                )
                continue

            print(f"Load model ckpt: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=trainer.device)
            if "model_state" in ckpt:
                model.load_state_dict(ckpt["model_state"])
                model.to(trainer.device)
                if decoder is not None and ckpt.get("decoder_state") is not None:
                    decoder.load_state_dict(ckpt["decoder_state"])
                    decoder.to(trainer.device)
            else:
                model.load_state_dict(ckpt)
                model.to(trainer.device)

            for test_ds in datasets:
                # keep previous behavior: evaluate self and ppicu (and cross only for ppicu)
                if args.pipeline in ["baseline", "llm"]:
                    if (
                        test_ds != "ppicu" and test_ds != train_ds
                    ):  # replicate earlier selection
                        continue
                # embedding pipeline evaluated on all in original embedding script
                print(
                    f"\n--- Evaluating test_ds={test_ds}, trained on {train_ds}, task={task}, rep={rep_str_for_model} ---"
                )

                if args.pipeline == "baseline":
                    args.embedding_model = (
                        ""  # ensure build_dataset chooses tabular datasets
                    )
                    rep_npz = rep_path_for_baseline(args.data_dir, test_ds, rep)
                    data_npz = labels_npz_path(args.data_dir, test_ds)
                    test_loader = get_loader(
                        rep_npz, data_npz, test_ds, "test", args, "all"
                    )
                elif args.pipeline == "llm":
                    rep_npz = rep_path_for_llm_rep(
                        args.rep_dir,
                        test_ds,
                        config["embedding_model"],
                        rep,
                        args.use_sum,
                        args.sum_model,
                    )
                    data_npz = labels_npz_path(args.data_dir, test_ds)
                    test_loader = get_loader(
                        rep_npz, data_npz, test_ds, "test", args, "all"
                    )
                else:
                    rep_npz = rep_path_for_embedding(
                        args.rep_dir, test_ds, args.embedding_model
                    )
                    data_npz = labels_npz_path(args.data_dir, test_ds)
                    test_loader = get_loader(
                        rep_npz, data_npz, test_ds, "test", args, args.embedding_model
                    )

                    # TSDE: align feature dim to training dataset's decoder expectation
                    if args.embedding_model == "TSDE":
                        D_train = get_embedding_input_dim("TSDE", train_ds)
                        test_loader.collate_fn = make_collate_fn(D_train)

                metrics = evaluate(
                    model,
                    decoder,
                    test_loader,
                    trainer,
                    task=task,
                    use_temporal=args.use_temporal,
                    ml_thresholding=args.ml_thresholding,
                )
                print("→ " + ", ".join(f"{k}={v}" for k, v in metrics.items()))


def finetune(datasets, reps, args, task):
    """
    Finetune only on ppicu as in previous scripts (cross-domain),
    skipping when test_ds == train_ds. Uses trainer.finetune_simple if present,
    else trainer.finetune.
    """
    for rep in reps:
        for train_ds in datasets:
            # Build config & model as in test_all
            if args.pipeline == "baseline":
                config = {
                    "dataset": train_ds,
                    "task": task,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "patience": args.patience,
                    "train_mask": "all",
                    "model_type": args.model_type,
                    "seed": args.seed,
                    "representation": rep,
                }
                rep_str_for_model = rep
                build_rep_path_for_test = lambda dst: rep_path_for_baseline(
                    args.data_dir, dst, rep
                )
                args.embedding_model = ""  # use tabular datasets for dataloaders

            elif args.pipeline == "llm":
                config = {
                    "dataset": train_ds,
                    "task": task,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "patience": args.patience,
                    "train_mask": "all",
                    "model_type": args.model_type,
                    "seed": args.seed,
                    "representation": (
                        f"all_{rep}" if not args.use_sum else f"all_sum_{rep}"
                    ),
                    "embed_dim": 4096,
                    "embedding_model": args.embedding_model or "Qwen3-Embedding-8B",
                    "summarization_model": args.sum_model,
                }
                rep_str_for_model = config["representation"]
                build_rep_path_for_test = lambda dst: rep_path_for_llm_rep(
                    args.rep_dir,
                    dst,
                    config["embedding_model"],
                    rep,
                    args.use_sum,
                    args.sum_model,
                )

            else:
                config = {
                    "dataset": train_ds,
                    "task": task,
                    "learning_rate": args.learning_rate,
                    "batch_size": args.batch_size,
                    "patience": args.patience,
                    "train_mask": "all",
                    "model_type": args.model_type,
                    "seed": args.seed,
                    "representation": args.embedding_model,
                    "embed_dim": get_embedding_input_dim(
                        args.embedding_model, train_ds
                    ),
                    "embedding_model": args.embedding_model,
                }
                rep_str_for_model = args.embedding_model
                build_rep_path_for_test = lambda dst: rep_path_for_embedding(
                    args.rep_dir, dst, args.embedding_model
                )

            model = initialize_model(
                args.model_type, task, train_ds, rep_str_for_model, args.embedding_model
            )
            trainer = Trainer(args, model, config)
            decoder = getattr(trainer, "decoder", None)

            ckpt_name = f"best_{trainer.experiment_string}.pt"
            ckpt_path = os.path.join(trainer.save_dir, ckpt_name)
            if not os.path.exists(ckpt_path):
                print(
                    f"[SKIP] No checkpoint for train_ds={train_ds}, rep={rep_str_for_model}: {ckpt_path}"
                )
                continue

            ckpt = torch.load(ckpt_path, map_location=trainer.device)
            if "model_state" in ckpt:
                model.load_state_dict(ckpt["model_state"])
                model.to(trainer.device)
                if decoder is not None and ckpt.get("decoder_state") is not None:
                    decoder.load_state_dict(ckpt["decoder_state"])
                    decoder.to(trainer.device)
            else:
                model.load_state_dict(ckpt)
                model.to(trainer.device)

            # Only finetune on ppicu, skip same-dataset
            for test_ds in datasets:
                if test_ds == train_ds:
                    continue
                if test_ds != "ppicu":
                    continue

                rep_npz = build_rep_path_for_test(test_ds)
                data_npz = labels_npz_path(args.data_dir, test_ds)

                # test loader
                test_loader = get_loader(
                    rep_npz, data_npz, test_ds, "test", args, "all"
                )
                # few-shot loader
                ft_loader = get_loader(
                    rep_npz, data_npz, test_ds, "finetune", args, "all"
                )

                # TSDE feature-align when embedding pipeline
                if args.pipeline == "embedding" and args.embedding_model == "TSDE":
                    D_train = get_embedding_input_dim("TSDE", train_ds)
                    test_loader.collate_fn = make_collate_fn(D_train)
                    ft_loader.collate_fn = make_collate_fn(D_train)

                # choose finetune API
                if hasattr(trainer, "finetune_simple"):
                    dec_state, model_state = trainer.finetune_simple(
                        model, decoder, ft_loader, test_loader
                    )
                    if decoder is not None and dec_state is not None:
                        decoder.load_state_dict(dec_state)
                    model.load_state_dict(model_state)
                else:
                    _, model_state = trainer.finetune(
                        model, decoder, ft_loader, test_loader
                    )
                    model.load_state_dict(model_state)

                model.to(trainer.device)
                if decoder is not None:
                    decoder.to(trainer.device)

                metrics = evaluate(
                    model,
                    decoder,
                    test_loader,
                    trainer,
                    task=task,
                    use_temporal=args.use_temporal,
                    ml_thresholding=args.ml_thresholding,
                )
                print(
                    f"Finetuned on {test_ds} (train {train_ds}, rep {rep_str_for_model}, k={args.finetune_size}) → "
                    + ", ".join(f"{k}={v}" for k, v in metrics.items())
                )


# ------------------------- CLI -------------------------
def add_common(p):
    p.add_argument(
        "--pipeline",
        choices=["embedding", "llm", "baseline"],
        required=True,
        help="embedding: single embedding model (TFM/TSDE/…); llm: prompt reps; baseline: mean/right/interp",
    )
    p.add_argument("--model_type", type=str, required=True)

    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument(
        "--task",
        choices=["forecast", "los", "mort", "lab", "feature", "gender", "age"],
        default="forecast",
    )
    p.add_argument("--data_dir", type=str, default="/project/shared/icu_datasets")
    p.add_argument("--rep_dir", type=str, default="/project/shared/health_embedding")

    # Embedding pipeline
    p.add_argument(
        "--embedding_model",
        type=str,
        default="",
        help="For pipeline=embedding: e.g., TFM, TSDE, Qwen3-Embedding-8B",
    )
    # LLM pipeline
    p.add_argument("--use_sum", action="store_true", default=False)
    p.add_argument(
        "--sum_model",
        type=str,
        default="medgemma-27b-text-it",
        help="Summarization model name when --use_sum",
    )
    # Baselines or LLM: reps to iterate (comma-separated)
    p.add_argument(
        "--reps",
        type=str,
        default="",
        help="Comma-list of reps. baseline: mean,right,interp. llm: zero_shot,ICD,Trend,CoT. "
        "embedding: typically leave empty (uses --embedding_model).",
    )

    p.add_argument("--use_temporal", action="store_true", default=False)
    p.add_argument(
        "--ml_thresholding",
        choices=["per_label", "prob"],
        default="per_label",
        help="Multilabel evaluation strategy.",
    )
    p.add_argument("--finetune", action="store_true", default=False)
    p.add_argument("--finetune_size", type=int, default=16)

    return p


def parse_reps(args):
    if args.pipeline == "embedding":
        if not args.embedding_model:
            raise ValueError("--embedding_model is required for pipeline=embedding")
        return [args.embedding_model]
    if args.reps:
        return [r.strip() for r in args.reps.split(",") if r.strip()]
    # sensible defaults for convenience
    if args.pipeline == "llm":
        return ["zero_shot", "ICD", "Trend", "CoT"]
    if args.pipeline == "baseline":
        return ["mean", "right", "interp"]
    return []


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    train_p = sub.add_parser("train")
    add_common(train_p)

    test_p = sub.add_parser("test")
    add_common(test_p)

    args = parser.parse_args()

    datasets = ["hirid", "mimic", "ppicu"]
    reps = parse_reps(args)
    task = args.task

    if args.mode == "train":
        train_all(datasets, reps, args, task)
    else:
        if args.finetune:
            finetune(datasets, reps, args, task)
        else:
            test_all(datasets, reps, args, task)


if __name__ == "__main__":
    main()
