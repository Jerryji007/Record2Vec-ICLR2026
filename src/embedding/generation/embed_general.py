import os
import pickle
import numpy as np
import torch
import time

import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import argparse
from tqdm import tqdm
from os.path import commonprefix

key_dict = {
    "sbf": "feature_text",
    "sbt": "time_text",
    "sbfc": "text",
    "all": "all_text",
}

# TO BE REMOVED when api branch merge
prompt_headings = {
    "zero_shot": (
        "You are a clinical agent that analyze and then provide the most concise summarization on ICU time series data for forecasting. "
        "Patient data:\n"
    ),
    "ICD": (
        "You are a clinical analysis agent. Summarize ICU time-series patient data for forecasting using this structure:\n"
        "Trend - overall direction of vitals, labs, therapies, and organ support.\n"
        "Seasonality - repeating cycles (e.g., circadian).\n"
        "Irregularities - acute deviations or events.\n\n"
        "Map each diagnosis to its affected organ system (cardiac, respiratory, hepatic, renal, neurologic, etc.). "
        "For every system, assign a severity score from 1 (least affected) to 10 (most severe) based on data patterns and level of support required.\n"
        "Output only the summary in clear clinical prose, concluding with a semicolon-separated list of organ systems and scores "
        "(e.g., “Cardiovascular 7/10; Respiratory 8/10; Hepatic 3/10”). Do not explain your reasoning. "
        "Patient data:\n"
    ),
    "Trend": (
        "Examine the data closely and describe the trend changes step by step over time. "
        "For example: from [start] to [midpoint], what happened? Then from [midpoint] to [end], what happened? "
        "After describing each phase, conclude with an overall summary in natural language. "
        "Summarize as many feature as possible starting from the most significant ones in concise words. Only include your description and summarization."
        "Patient data:\n"
    ),
    "CoT": (
        "You are a healthcare agent that summarizes ICU patients' time series status for future time series forecasting. "
        "Analyze this step by step. \nStep 1: Analyze the time series data to identify key trends. \n"
        "Step 2: Based on the identified trends, determine potential clinical implications. \n"
        "Step 3: Summarize the findings and suggest possible interventions. \n"
        "Summarize as many feature as possible starting from the most significant ones in concise words and only respond with your summarization. "
        "Patient data:\n"
    ),
    "None": "",
}

all_texts = list(prompt_headings.values())
shared = commonprefix(all_texts)

suffixes = {k: v[len(shared) :] for k, v in prompt_headings.items()}


class TextDataset(Dataset):
    def __init__(self, texts, heading_type, tokenizer, max_length=32000):
        self.texts = texts
        self.heading_type = heading_type
        self.tokenizer = tokenizer

        # max_length configured by model type
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Use commone prefix in headings from predefined set as instructio
        prefix = f"Instruct: {shared}\nQuery: {suffixes[self.heading_type]}"
        return prefix + self.texts[idx]

    def collate_fn(self, batch):
        # Tokenize the entire batch at once with padding
        results = self.tokenizer(
            batch,
            padding=True,  # pad to longest in batch
            truncation=True,  # do not truncate
            max_length=self.max_length,  # new max length configuration
            return_tensors="pt",
        )
        return results


def tokenize_text(tokenizer, texts):
    """
    tokenize text in batches
    """
    start_time = time.time()
    print("Start tokenizing")
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    end_time = time.time()

    print(f"Tokenizing takes {end_time - start_time}s")

    return inputs


def load_text_mimic(text_dict, text_type):
    texts = text_dict[key_dict[text_type]]

    assert (len(k) == 71 for k in texts)

    texts_merged = []

    for stay in texts:
        texts_merged.extend(stay)

    return texts_merged


def load_text_hirid(text_dict, text_type):
    texts = text_dict["text"]

    assert (len(k) == 66 for k in texts)

    texts_merged = []

    for stay in texts:
        texts_merged.extend(stay)

    return texts_merged


def embed_text_general_model(model, b_input_ids, b_attention_mask):
    outputs = model(
        b_input_ids, attention_mask=b_attention_mask, output_hidden_states=True
    )

    last_hidden_layer = outputs.hidden_states[-2].mean(dim=1).squeeze()

    return last_hidden_layer


def embed_text_embedding_model(model, input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    last_hidden = outputs.last_hidden_state

    # compute index of last non‑pad token for each batch element
    seq_lens = attention_mask.sum(dim=1) - 1
    batch_idx = torch.arange(0, last_hidden.size(0), device=last_hidden.device)

    # gather the last-token representations
    pooled = last_hidden[batch_idx, seq_lens]

    # L2‑normalize
    return F.normalize(pooled, p=2, dim=1)


def main(args):
    dataset = args.dataset
    model_path = args.model
    output_dir = args.output_dir
    dataset_root = args.dataset_root
    first_half = args.first_half
    split_half = args.split_half
    heading_type = args.heading_type
    embed_sum = args.embed_sum
    sum_model = args.sum_model

    if first_half:
        split_half = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"Start embedding on\n"
        f"  Dataset: {dataset}\n"
        f"  Model: {args.model}\n"
        f"  Device: {device}"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_path == "sentence-transformers/all-MiniLM-L6-v2":
        model = AutoModel.from_pretrained(model_path, device_map="auto")
    elif model_path == "/model-weights/Qwen3-Embedding-8B":
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",  # Fully load weights into memory.
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",  # Fully load weights into memory.
        )

    max_length = {
        "/model-weights/Qwen3-Embedding-8B": 32000,
        "sentence-transformers/all-MiniLM-L6-v2": 512,
    }[model_path]

    torch.cuda.empty_cache()

    print(f"Load model from {model_path} succed!")

    if embed_sum:
        if sum_model:
            text_data_path = os.path.join(
                dataset_root,
                f"{dataset}/{dataset}_{sum_model}_summaries_{heading_type}.pkl",
            )
        else:
            text_data_path = os.path.join(
                dataset_root, f"{dataset}/{dataset}_summarization_{heading_type}.pkl"
            )
    else:
        text_data_path = os.path.join(
            dataset_root, f"{dataset}/{dataset}_sbf_short.pkl"
        )

    start = time.time()
    with open(text_data_path, "rb") as f:
        merged_text = pickle.load(f)
    print(f"Text loaded from pickle {text_data_path} in {time.time() - start:.2f}s")

    # if dataset == "hirid":
    #     if text_type != "all":
    #         merged_text = load_text_hirid(text_dict, text_type)
    #     else:
    #         merged_text = text_dict["text"]

    # else:
    #     if text_type != "all":
    #         merged_text = load_text_mimic(text_dict, text_type)
    #     else:
    #         merged_text = text_dict["all_text"]

    # merged_text = merged_text[:256]

    print(f"Merged text has a total of {len(merged_text)} lines")

    if split_half:
        first_half_length = len(merged_text) // 2
        if args.first_half:
            merged_text = merged_text[:first_half_length]
        else:
            merged_text = merged_text[first_half_length:]

    print(f"Truncated text has a total of {len(merged_text)} lines")

    if embed_sum:
        dataset = TextDataset(merged_text, "None", tokenizer, max_length)
    else:
        dataset = TextDataset(merged_text, heading_type, tokenizer, max_length)

    batch_size = 8
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn
    )

    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if model_path in ["/model-weights/Qwen3-Embedding-8B"]:
                last_hidden_layer = embed_text_embedding_model(
                    model, input_ids, attention_mask
                )
            else:
                last_hidden_layer = embed_text_general_model(
                    model, input_ids, attention_mask
                )

            all_embeddings.append(last_hidden_layer.cpu().numpy())

    all_embeddings = np.vstack(all_embeddings)

    dataset_name = args.dataset
    model_name = "Qwen3-Embedding-8B"
    if split_half:
        if first_half:
            output_path = os.path.join(
                output_dir, f"{dataset_name}_all_{model_name}_{heading_type}_1.npz"
            )
        else:
            output_path = os.path.join(
                output_dir, f"{dataset_name}_all_{model_name}_{heading_type}_1.npz"
            )
    else:
        if embed_sum:
            output_path = os.path.join(
                output_dir,
                f"{dataset_name}_all_sum_{sum_model}_{model_name}_{heading_type}.npz",
            )
        else:
            output_path = os.path.join(
                output_dir, f"{dataset_name}_all_{model_name}_{heading_type}.npz"
            )

    print(f"Emebdding shapeds as {all_embeddings.shape} saved to {output_path}")

    np.savez(output_path, embeddings=all_embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute embeddings using the Llama-3.2-1B model directly, processing each text sequentially."
    )
    parser.add_argument(
        "--dataset",
        default="ppicu",
        help="Dataset name (default: ppicu)",
    )
    parser.add_argument(
        "--model",
        default="/model-weights/Qwen3-Embedding-8B",
        help="Gemma model checkpoint (default: '/model-weights/Qwen3-Embedding-8B')",
    )
    parser.add_argument(
        "--output_dir",
        default="/project/shared/health_embedding",
        help="Output directory for NPZ files",
    )
    parser.add_argument(
        "--dataset_root",
        default="/project/shared/health_summarization",
        help="Root path for dataset pickle files",
    )

    parser.add_argument(
        "--split_half", action="store_true", help="split merged text in half"
    )

    parser.add_argument(
        "--first_half",
        action="store_true",
        help="store the first half of the embedding",
    )

    parser.add_argument("--text_type", default="sbf", help="configure the text type")

    parser.add_argument(
        "--heading_type",
        default="ICD",
        choices=["zero_shot", "ICD", "Trend", "CoT"],
        help="configure the heading type",
    )
    parser.add_argument(
        "--embed_sum", action="store_true", help="run summarization embedding"
    )
    parser.add_argument("--sum_model", default="", help="Summarization model")

    args = parser.parse_args()
    main(args)
