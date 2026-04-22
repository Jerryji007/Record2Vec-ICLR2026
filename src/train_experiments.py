#!/usr/bin/env python3
import argparse
import re
import torch
import random
import numpy as np
from torch.utils.data import DataLoader

# Original dataset imports, now including LLMDataset classes from the same module.
from data_loaders.datasets.dummy import DummyTSDataset
from data_loaders.datasets.hirid import HiridDataset, HiridLLMDataset
from data_loaders.datasets.mimic import MimicDataset, MimicLLMDataset
from data_loaders.datasets.ppicu import PpicuDataset, PpicuLLMDataset
from exp.task_trainer import Trainer
from torch.utils.data import WeightedRandomSampler


def check_representation(rep: str) -> str:
    """
    Validate the representation method.
    Valid options:
      - Non-LLM: 'mean', 'right', 'interp'
      - LLM-based: 'all', 'sbf', 'sbt'
    """
    valid_options = {"mean", "right", "interp", "all", "sbf", "sbt", "sbfc"}
    if rep in valid_options:
        return rep
    raise argparse.ArgumentTypeError(
        f"Invalid representation method '{rep}'. Must be one of {valid_options}."
    )


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_model(model_str, task, dataset, representation, llm_type):
    """
    Initialize and return a model instance.

    For LLM-based representations ('all', 'sbt', 'sbf') the input dimensions (D1)
    and sequence lengths (T) are adjusted based on the dataset and llm_type.
    For non-LLM representations ('mean', 'right', 'interp'), default dataset parameters are used.

    The rules for LLM-based adjustments are:
      - For "all":   D1 = llm_emb_dim, T = 8.
      - For "sbt":   D1 = llm_emb_dim, T remains as default.
      - For "sbf":   D1 remains as default, T = llm_emb_dim.
    In all cases, T2 remains unchanged and the forecast output dimension (D2) is fixed as 40.
    """
    # Default parameters for non-LLM representations.
    dataset_params = {
        "mimic": {"D": 60, "T": 48, "T2": 24, "Lab": 21},
        "hirid": {"D": 64, "T": 48, "T2": 24, "Lab": 24},
        "ppicu": {"D": 75, "T": 48, "T2": 24, "Lab": 25},
        "eicu": {"D": 50, "T": 48, "T2": 24},
        "dummy": {"D": 61, "T": 48, "T2": 24},
    }
    if dataset not in dataset_params:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Create a dictionary for LLM embedding dimensions.
    llm_emb_dims = {
        "sentence-transformers_all-MiniLM-L6-v2": 384,
        "gemma": 4096,
        "gemini-2.0-flash": 768,
        "Qwen2-7B": 3584,
        "Qwen3-Embedding-8B": 4096,
        "TFM": 1280,
        "Qwen3-Embedding-8B+TFM": 5376,
    }
    default_D1 = dataset_params[dataset]["D"]
    default_T = dataset_params[dataset]["T"]
    T2 = dataset_params[dataset]["T2"]

    # Adjust parameters if using an LLM-based representation for supported datasets.
    if "all" in representation and dataset in {
        "hirid",
        "mimic",
        "ppicu",
    }:
        emb_dim = llm_emb_dims.get(llm_type)
        if emb_dim is None:
            raise ValueError(f"Unknown LLM type: {llm_type}")
        if "all" in representation:
            D1 = 75
            T = default_T
        elif "sbt" in representation:
            D1 = emb_dim
            T = default_T
        elif "sbf" in representation:
            D1 = default_D1
            T = emb_dim
    elif llm_type == "TSDE":
        D1 = 75
        T = default_T
    elif llm_type == "TFM":
        D1 = 75
        T = default_T
    else:
        # Use default parameters.
        D1 = default_D1
        T = default_T

    if task == "forecast" or model_str == "timemixer":
        D2 = dataset_params[dataset]["D"]
    if task == "feature":
        D2 = 2
    if task == "lab":
        D2 = dataset_params[dataset]["Lab"]

    # Determine the model class suffix based on the task.
    if task == "forecast":
        suffix = "Forecast"
    elif task in ["los", "kf", "age"]:
        suffix = "Regression"
    elif task in ["mort", "sepsis", "aki", "AKI", "gender"]:
        suffix = "Classification"
    elif task in ["lab", "feature"]:
        suffix = "MultiLabelClassification"
    else:
        raise ValueError(f"Unknown task: {task}")

    # Map model string to proper prefix.
    model_prefix = {
        "mlp": "MLP",
        "lstm": "LSTM",
        "patchtsmixer": "PatchTSMixer",
        "timemixer": "TimeMixer",
    }
    if model_str not in model_prefix:
        raise ValueError(f"Unknown model type: {model_str}")
    prefix = model_prefix[model_str]
    class_name = prefix + suffix  # e.g., "MLPForecast" or "LSTMRegression"

    # Dynamically import the module from the models directory.
    module_name = f"models.{model_str}"
    try:
        module = __import__(module_name, fromlist=[class_name])
    except ImportError as e:
        raise ImportError(f"Could not import module {module_name}") from e

    try:
        model_class = getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(
            f"Module {module_name} does not have a class named {class_name}"
        ) from e

    # For forecast tasks or if using timemixer, include T2 and D2.
    if task == "forecast" or model_str == "timemixer":
        model = model_class(D1, T, D2, T2)
        print(
            f"Initialized model: {class_name} with D1={D1}, T={T}, D2={D2}, T2={T2} for dataset {dataset}, task {task}, representation {representation}, llm_type {llm_type}"
        )
    elif task in ["lab", "feature"]:
        model = model_class(D1, T, num_labels=D2)
        print(
            f"Initialized model: {class_name} with D1={D1}, T={T}, num_labels={D2} "
            f"for dataset {dataset}, task {task}, representation {representation}, llm_type {llm_type}"
        )
    else:
        model = model_class(D1, T)
        print(
            f"Initialized model: {class_name} with D1={D1}, T={T} for dataset {dataset}, task {task}, representation {representation}, llm_type {llm_type}"
        )

    return model


def configure_task(args):
    """
    Create a task configuration dictionary based on command-line arguments.
    """
    task_config = {
        "dataset": args.dataset,
        "task": args.task,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "train_mask": args.train_mask,
        "model_type": args.model,
        "seed": args.seed,
        "representation": args.representation,
        "llm_type": args.llm_type,
    }
    print("Task configuration:", task_config)
    return task_config


def main(args):
    # Print the configuration
    print("Training Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Task: {args.task}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Seed: {args.seed}")
    print(f"  Representation Method: {args.representation}")
    print(f"  LLM Type: {args.llm_type}")
    print(f"  Patience: {args.patience}")

    # Set seed for reproducibility.
    set_seed(args.seed)

    # Define the base path for all datasets.
    base_path = "/project/shared/icu_datasets/"

    # Dataset loading
    if args.dataset == "dummy":
        print("Using dummy dataset.")
        train_dataset = DummyTSDataset(
            split="train", train_mask=args.train_mask, seed=args.seed
        )
        val_dataset = DummyTSDataset(split="val", seed=args.seed)
    elif args.dataset in ["hirid", "mimic", "ppicu"]:
        # Check if using an LLM-based representation.
        if args.representation in {"all", "sbt", "sbf"}:
            # The labels are stored in the 'mean' file.
            datapath = f"{base_path}{args.dataset}/{args.dataset}_mean_with_lab.npz"
            rep_datapath = (
                f"{base_path}{args.dataset}_{args.representation}_{args.llm_type}.npz"
            )
            print(
                f"Loading LLM-based {args.dataset} dataset from {rep_datapath} (labels from {datapath})"
            )
            if args.dataset == "hirid":
                train_dataset = HiridLLMDataset(
                    rep_datapath,
                    datapath,
                    "train",
                    args.llm_type,
                    args.representation,
                    args.train_mask,
                    args.seed,
                )
                val_dataset = HiridLLMDataset(
                    rep_datapath,
                    datapath,
                    "val",
                    args.llm_type,
                    args.representation,
                    args.train_mask,
                    args.seed,
                )
            elif args.dataset == "mimic":
                train_dataset = MimicLLMDataset(
                    rep_datapath,
                    datapath,
                    "train",
                    args.llm_type,
                    args.representation,
                    args.train_mask,
                    args.seed,
                )
                val_dataset = MimicLLMDataset(
                    rep_datapath,
                    datapath,
                    "val",
                    args.llm_type,
                    args.representation,
                    args.train_mask,
                    args.seed,
                )
            elif args.dataset == "ppicu":
                train_dataset = PpicuLLMDataset(
                    rep_datapath,
                    datapath,
                    "train",
                    args.llm_type,
                    args.representation,
                    args.train_mask,
                    args.seed,
                )
                val_dataset = PpicuLLMDataset(
                    rep_datapath,
                    datapath,
                    "val",
                    args.llm_type,
                    args.representation,
                    args.train_mask,
                    args.seed,
                )
        else:
            # For non-LLM representations.
            data_path = (
                f"{base_path}{args.dataset}/{args.dataset}_{args.representation}.npz"
            )
            y_path = f"{base_path}{args.dataset}/{args.dataset}_mean_with_lab.npz"
            print(f"Loading {args.dataset} dataset from {data_path}")
            if args.dataset == "hirid":
                train_dataset = HiridDataset(
                    data_path, y_path, "train", args.train_mask, args.seed
                )
                val_dataset = HiridDataset(
                    data_path, y_path, "val", args.train_mask, args.seed
                )
            elif args.dataset == "mimic":
                train_dataset = MimicDataset(
                    data_path, y_path, "train", args.train_mask, args.seed
                )
                val_dataset = MimicDataset(
                    data_path, y_path, "val", args.train_mask, args.seed
                )
            elif args.dataset == "ppicu":
                train_dataset = PpicuDataset(
                    data_path, y_path, "train", args.train_mask, args.seed
                )
                val_dataset = PpicuDataset(
                    data_path, y_path, "val", args.train_mask, args.seed
                )
            else:
                raise ValueError(f"Dataset {args.dataset} is not supported.")
    else:
        # For other datasets like ppicu or eicu, assuming only non-LLM representations.
        data_path = f"{base_path}{args.dataset}_{args.representation}.npz"
        print(f"Loading {args.dataset} dataset from {data_path}")
        # Here you would call the corresponding dataset loader.
        train_dataset = load_dataset(
            args.dataset
        )  # Replace with actual implementation.
        val_dataset = load_dataset(args.dataset)

    # Initialize DataLoaders with the provided batch size.
    # After you load your train_dataset, add sampler only if task is "mort"
    if args.task == "mort":
        # Assuming each sample is a tuple and the mortality label is at index 3
        # Extract the labels from the dataset
        targets = [int(sample[3]) for sample in train_dataset]

        # Compute the number of samples for each class
        class_counts = np.bincount(targets)
        # Compute weights: the weight for a class is the inverse of its count
        weights_per_class = 1.0 / class_counts

        # Assign a weight to each sample based on its class
        sample_weights = [weights_per_class[label] for label in targets]

        # Create the weighted random sampler
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
        print("training with sampler")
        # Use the sampler in the DataLoader (do not set shuffle)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler
        )
    else:
        # Default DataLoader with shuffling for other tasks
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model and task configuration.
    print(
        f"Initializing model: {args.model}, task: {args.task}, dataset: {args.dataset}"
    )
    model = initialize_model(
        args.model, args.task, args.dataset, args.representation, args.llm_type
    )
    task_config = configure_task(args)

    print(
        f"Using representation method: {args.representation} with llm_type: {args.llm_type}"
    )

    # Initialize Trainer with arguments, model, and task configuration.
    trainer = Trainer(args, model, task_config)

    # Begin training using train_loader and val_loader.
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a specific model on a specific ICU dataset for a specific task."
    )

    # Model, Dataset, Task arguments
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "lstm", "patchtsmixer", "timemixer"],
        default="mlp",
        help="Model type to train (default: mlp).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["dummy", "mimic", "hirid", "ppicu", "eicu"],
        default="ppicu",
        help="Dataset to use (default: dummy).",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["forecast", "los", "mort", "lab", "feature", "gender", "age"],
        default="forecast",
        help="Task to perform (default: forecast).",
    )

    parser.add_argument(
        "--train_mask",
        type=str,
        choices=["all", "half", "quarter"],
        default="all",
        help="Mask for training data (default: all).",
    )

    # Hyperparameters
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for training (default: 1e-4).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for training (default: 1024).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed for reproducibility (default: 42).",
    )

    # Representation method with custom validation.
    parser.add_argument(
        "--representation",
        type=check_representation,
        choices=["mean", "right", "interp", "all", "sbf", "sbt"],
        default="mean",
        help=(
            "Representation method. Non-LLM: mean, right, interp; "
            "LLM-based: all, sbf, sbt. (default: mean)"
        ),
    )

    parser.add_argument(
        "--llm_type",
        type=str,
        default="Qwen3-Embedding-8B",
        help="LLM type for LLM-based representations (default: sentence-transformers_all-MiniLM-L6-v2).",
    )

    # Patience argument for early stopping.
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Number of epochs to wait for improvement in validation loss before early stopping (default: 8).",
    )

    parser.add_argument(
        "--gpu", type=int, default=1, help="GPU device to use (default: 0)."
    )

    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of epochs to train."
    )

    args = parser.parse_args()
    main(args)
