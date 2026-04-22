#!/usr/bin/env python
# extract_timesfm_embeddings_v2.py
"""
Extract TimesFM embeddings - implementation mirrors the official
github.com/google-research/timesfm usage example.
"""

import argparse, math, numpy as np, torch
import os
import timesfm

from tqdm import tqdm
from pathlib import Path


# ----------------------------------------------------------------------
def pad_to_multiple(arr: np.ndarray, m: int) -> np.ndarray:
    pad = (-len(arr)) % m
    return np.pad(arr, (0, pad), mode="constant")


def zscore(a: np.ndarray, axis=None, eps: float = 1e-8) -> np.ndarray:
    """
    Standard score normalisation along the given axis.
      a      : input array
      axis   : None (flatten) or int/tuple; matches np.mean / np.std signature
      eps    : tiny constant to avoid /0
    Returns array of same shape as `a`.
    """
    mu = a.mean(axis=axis, keepdims=True)
    sig = a.std(axis=axis, keepdims=True)
    return (a - mu) / (sig + eps)


# ----------------------------------------------------------------------
def build_series(x_all: np.ndarray, patch_len: int) -> torch.Tensor:
    """
    z-score per feature ➜ mean over 75 vars ➜ pad ➜ torch tensor
    """
    out = []
    for sample in x_all:  # sample (48, 75)
        uni = zscore(sample, axis=0).mean(axis=-1)  # (48,)
        out.append(pad_to_multiple(uni, patch_len))  # (96,)
    return torch.tensor(np.stack(out), dtype=torch.float32)


# ----------------------------------------------------------------------
def main(args):
    device_str = "gpu" if torch.cuda.is_available() else "cpu"

    print("→ loading data …")
    npz = np.load(
        os.path.join(args.infolder, f"{args.dataset}/{args.dataset}_mean.npz")
    )
    x_all = npz["all_x"]  # (N,48,75)
    x_masks = npz["all_x_mask"]

    print("→ building TimesFM model …")
    tfm = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend=device_str,
            per_core_batch_size=args.batch,
            horizon_len=24,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=50,
            model_dims=1280,
            use_positional_embedding=False,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            huggingface_repo_id="google/timesfm-2.0-500m-pytorch"
        ),
    )
    tfm._model.eval()
    dev = tfm._device

    N, T, F = x_all.shape
    pad_len = 64

    series_np = x_all.transpose(0, 2, 1).reshape(N * F, T)
    valid_np = x_masks.transpose(0, 2, 1).reshape(N * F, T) > 0

    series = torch.tensor(series_np, dtype=torch.float32, device=tfm._device)
    valid = torch.tensor(valid_np, dtype=torch.bool, device=tfm._device)

    T = 48  # context hours
    H = 24  # forecast horizon
    pad_ctx = 64  # next multiple of 32 for the context
    seq_len = pad_ctx + H  # 88
    M = N * F  # total rows after feature‑flattening

    ctx_x = torch.zeros(M, pad_ctx, device=dev, dtype=torch.float32)
    ctx_x[:, -T:] = series

    # padding mask (True = ignore)
    paddings = torch.ones(M, seq_len, dtype=torch.float32, device=dev)
    paddings[:, -(T + H) : -H] = (~valid).float()  # honour missing values
    paddings[:, -H:] = 1.0

    freq_mat = torch.zeros(M, 1, dtype=torch.long, device=dev)

    print("→ decoding …")
    embeds_flat = np.empty((M, tfm.hparams.model_dims), dtype=np.float32)
    step = args.batch
    with torch.no_grad():
        for s in tqdm(range(0, M, step)):
            e = min(s + step, M)

            in_ts = ctx_x[s:e]
            pad = paddings[s:e]
            freq_b = freq_mat[s:e]

            _, _, hiddens = tfm._model.decode(
                input_ts=in_ts,  # (B, 64)
                paddings=pad,  # (B, 64)  True = ignore
                freq=freq_b,  # (B, 64)  0 = high‑freq
                horizon_len=24,
                output_patch_len=tfm.hparams.output_patch_len,
                return_forecast_on_context=False,
                return_hidden_states=True,  # returns tuple of tensors
            )
            # penultimate layer, context tokens only
            patch_len = 32
            n_patches = pad_ctx // patch_len

            step_mask = pad[:, :pad_ctx]

            patch_valid = step_mask.view(-1, n_patches, patch_len).sum(-1) == 0
            patch_valid = patch_valid.unsqueeze(-1).float()

            penult = hiddens[-2].mean(dim=1)
            embeds_flat[s:e] = penult.cpu().numpy()

    print(f"→ saving embeddings to {args.outfile}")
    embeds_per_feat = embeds_flat.reshape(N, F, -1)
    embeds_series = embeds_per_feat.mean(axis=1)
    np.savez_compressed(
        os.path.join(args.outfile, f"{args.dataset}/{args.dataset}_TFM"), embeds_series
    )
    print("✓ done saved", embeds_series.shape)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--infolder", default="/project/shared/icu_datasets")
    ap.add_argument("--outfile", default="/project/shared/health_embedding")
    ap.add_argument("--dataset", default="ppicu")
    ap.add_argument("--batch", type=int, default=1024)
    main(ap.parse_args())
