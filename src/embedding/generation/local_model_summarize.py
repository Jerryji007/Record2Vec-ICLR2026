#!/usr/bin/env python
"""
Summarise ICU notes with one of:
  • Meta-Llama-3-8B-Instruct   (causal LM, single-GPU)
  • Google MedGemma-27B-text-it (causal LM, *model-parallel*)

New feature
-----------
Add `--num_parts` and `--part` so large datasets can be processed in
multiple independent jobs (e.g. Slurm array).  Part *k* (0-indexed)
receives roughly len(notes)/num_parts examples.
"""

import argparse, math, pickle
from pathlib import Path
from itertools import islice

import torch
from tqdm.auto import tqdm
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
)

import torch, torch._dynamo as dynamo

dynamo.config.cache_size_limit = 256

from utils.prompt_headings import prompt_headings

DATA_ROOT = Path("/project/shared/icu_raw_text")
OUT_ROOT = Path("/project/shared/health_summarization")


def batched(iterable, n):
    it = iter(iterable)
    while chunk := list(islice(it, n)):
        yield chunk


def _prepare_tokenizer(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    return tok


def build_model_parallel_pipe(model_path: str, batch_size: int = 4):
    tok = _prepare_tokenizer(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        batch_size=batch_size,
    )


def summarise_with_pipe(pipe, notes, heading_key: str, batch_size: int = 4):
    sys_prompt = prompt_headings[heading_key]
    summaries = []
    for batch in tqdm(
        batched(notes, batch_size),
        total=math.ceil(len(notes) / batch_size),
        desc="Batched summarisation",
        unit="batch",
        dynamic_ncols=True,
    ):
        messages_batch = [
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": note},
            ]
            for note in batch
        ]
        outs = pipe(messages_batch, max_new_tokens=512, do_sample=False)
        for single_out in outs:
            assistant_msg = single_out[0]["generated_text"][-1]
            summaries.append(assistant_msg["content"].strip())
    return summaries


def summarise_longest(pipe, notes, heading_key, batch_size):
    order = sorted(range(len(notes)), key=lambda i: len(notes[i]), reverse=True)
    sorted_notes = [notes[i] for i in order]

    sorted_summaries = summarise_with_pipe(pipe, sorted_notes, heading_key, batch_size)

    summaries = [None] * len(notes)
    for idx, summ in zip(order, sorted_summaries):
        summaries[idx] = summ
    return summaries


def slice_to_part(notes, part: int, num_parts: int):
    """Return the slice of `notes` assigned to `part` (0-indexed)."""
    if not (0 <= part < num_parts):
        raise ValueError(f"--part must be in [0, {num_parts-1}]")
    total = len(notes)
    part_size = math.ceil(total / num_parts)
    start = part * part_size
    end = min(start + part_size, total)
    return notes[start:end]


def main(args):
    data_file = DATA_ROOT / f"{args.dataset}_sbf_short.pkl"
    notes = pickle.load(open(data_file, "rb"))
    if not args.run_full:
        notes = sorted(notes, key=len, reverse=True)[:5000]

    # Select the slice for this job
    notes = slice_to_part(notes, args.part, args.num_parts)
    print(f"Processing part {args.part+1}/{args.num_parts}: {len(notes)} notes")

    pipe = build_model_parallel_pipe(args.model_path, args.batch_size)
    summaries = summarise_longest(pipe, notes, args.headings, args.batch_size)

    out_dir = OUT_ROOT / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = Path(args.model_path).name
    out_file = out_dir / (
        f"{args.dataset}_{tag}_summaries_{args.headings}"
        f"_part{args.part}of{args.num_parts}.pkl"
    )
    pickle.dump(summaries, open(out_file, "wb"))
    print(f"Saved {len(summaries)} summaries -> {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarise ICU notes with LLaMA-3 or MedGemma (model-parallel)."
    )
    parser.add_argument(
        "--headings", choices=["zero_shot", "ICD", "Trend", "CoT"], default="ICD"
    )
    parser.add_argument(
        "--dataset", choices=["ppicu", "hirid", "mimic"], default="ppicu"
    )
    parser.add_argument(
        "--run_full", action="store_true", help="Process full dataset (else first 5000)"
    )
    parser.add_argument(
        "--model_path", required=True, help="Checkpoint dir: LLaMA-3 / MedGemma"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Generation batch size"
    )

    # NEW FLAGS
    parser.add_argument(
        "--num_parts",
        type=int,
        default=7,
        help="Total number of equal parts to split the dataset",
    )
    parser.add_argument(
        "--part",
        type=int,
        default=0,
        help="Index (0-based) of the part this run should handle",
    )

    args = parser.parse_args()
    main(args)
