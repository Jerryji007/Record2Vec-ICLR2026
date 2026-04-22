import os
import re
import csv
import torch
import argparse
from torch.utils.data import DataLoader
import torchmetrics

# Import the dataset loaders.
from data_loaders.datasets.dummy import DummyTSDataset
from data_loaders.datasets.hirid import HiridDataset, HiridLLMDataset
from data_loaders.datasets.ppicu import PpicuDataset, PpicuLLMDataset
from data_loaders.datasets.mimic import MimicDataset, MimicLLMDataset

# Import the shared initialize_model function from the training script.
# (Adjust the module path if needed.)
from train_experiments import initialize_model

# Base path for dataset files (should match the training script).
BASE_PATH = ""

class Tester:
    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        self.task = args.task
        # Directory where model checkpoints are saved.
        self.runs_dir = os.path.join("", self.dataset, self.task)
        device = (
            torch.device(f"cuda:{self.args.gpu}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.device = device
        # num features
        print(f"Using device: {self.device}")

        if not os.path.exists(self.runs_dir):
            raise ValueError(f"Runs directory {self.runs_dir} does not exist.")

    def parse_model_filename(self, filename):
        """
        Parses a filename of the form:
        For normal representations:
            best_{model}_{representation}_{train_mask}_{lr}_{batch_size}_seed_{seed}.pt
        For LLM-based representations:
            best_{model}_{representation}_{llm_type}_{train_mask}_{lr}_{batch_size}_seed_{seed}.pt
        where the llm_type string may itself contain underscores.

        Returns a dictionary with the following keys:
        model, representation, llm_type, train_mask, lr, bs, seed
        """
        # Remove "best_" prefix and ".pt" suffix.
        base = filename.replace("best_", "").replace(".pt", "")

        # Split off the seed part (assumes '_seed_' is present)
        if "_seed_" not in base:
            print(
                f"Filename {filename} does not contain '_seed_' and cannot be parsed."
            )
            return None
        base_without_seed, seed_str = base.rsplit("_seed_", 1)
        seed = seed_str

        # Now, the beginning of base_without_seed has model and representation.
        # Do a left-split into three parts so that the first two fields are isolated.
        parts = base_without_seed.split("_", 2)
        if len(parts) < 3:
            print(f"Filename {filename} does not match expected pattern.")
            return None
        model = parts[0]
        representation = parts[1]
        remainder = parts[2]  # This contains either:
        #   (a) {train_mask}_{lr}_{bs} for normal representations, or
        #   (b) {llm_type}_{train_mask}_{lr}_{bs} for LLM-based representations.

        # Right-split the remainder into the last three tokens.
        tail_tokens = remainder.rsplit("_", 3)
        if len(tail_tokens) == 4:
            llm_type = tail_tokens[0]
            train_mask = tail_tokens[1]
            lr = tail_tokens[2]
            bs = tail_tokens[3]
        elif len(tail_tokens) == 3:
            llm_type = None
            train_mask, lr, bs = tail_tokens
        else:
            print(f"Filename {filename} does not match expected pattern.")
            return None
        self.args.seed = int(seed)
        return {
            "model": model,
            "representation": representation,
            "llm_type": llm_type,
            "train_mask": train_mask,
            "lr": lr,
            "bs": bs,
            "seed": seed,
        }

    def build_test_dataset(self, representation, llm_type):
        """
        Build and return the test dataset (and its DataLoader) according to the dataset,
        representation type, and llm_type.
        """
        # For dummy dataset.
        if self.dataset == "dummy":
            test_dataset = DummyTSDataset(split="test", seed=self.args.seed)
        # For hirid and mimic, check if using LLM-based representation.
        elif self.dataset in ["hirid", "mimic", "ppicu"]:
            if representation in {"all", "sbt", "sbf"}:
                # LLM-based: labels are in the 'mean' file.
                datapath = f"{BASE_PATH}{self.dataset}_mean.npz"
                rep_datapath = (
                    f"{BASE_PATH}{self.dataset}_{representation}_{llm_type}.npz"
                )
                print(
                    f"Loading LLM-based {self.dataset} test dataset from {rep_datapath} (labels from {datapath})"
                )
                if self.dataset == "hirid":
                    test_dataset = HiridLLMDataset(
                        rep_datapath,
                        datapath,
                        "test",
                        llm_type,
                        representation,
                        "all",
                        self.args.seed,
                    )
                elif self.dataset == "mimic":
                    test_dataset = MimicLLMDataset(
                        rep_datapath,
                        datapath,
                        "test",
                        llm_type,
                        representation,
                        "all",
                        self.args.seed,
                    )
                elif self.dataset == "ppicu":
                    test_dataset = PpicuLLMDataset(
                        rep_datapath,
                        datapath,
                        "test",
                        llm_type,
                        representation,
                        "all",
                        self.args.seed,
                    )
            else:
                # Non-LLM representation.
                data_path = f"{BASE_PATH}{self.dataset}_{representation}.npz"
                print(f"Loading {self.dataset} test dataset from {data_path}")
                if self.dataset == "hirid":
                    test_dataset = HiridDataset(
                        data_path, "test", "all", self.args.seed
                    )
                    # print(len(test_dataset))
                    # validation_dataset = HiridDataset(
                    #     data_path, "val", "half", self.args.seed
                    # )
                    # print(len(validation_dataset))
                    # assert 1 == 2
                elif self.dataset == "mimic":
                    test_dataset = MimicDataset(
                        data_path, "test", "all", self.args.seed
                    )
                elif self.dataset == "ppicu":
                    test_dataset = PpicuDataset(
                        data_path, "test", "all", self.args.seed
                    )
        # For other datasets like ppicu, eicu, etc.
        elif self.dataset in ["eicu"]:
            data_path = f"{BASE_PATH}{self.dataset}_{representation}.npz"
            # Replace the following with your actual dataset loader.
            from data_loaders.datasets.dummy import DummyTSDataset

            test_dataset = DummyTSDataset(split="test", seed=self.args.seed)
            print(
                f"Loading {self.dataset} test dataset from {data_path} (using dummy loader as placeholder)"
            )
        else:
            raise ValueError(
                f"Dataset {self.dataset} is not supported in test dataset loading."
            )
        return test_dataset

    def get_targets(self, batch):
        """
        Given a batch, returns the target tensor for the current task.
        Batch indices:
          index 0: inputs
          index 1: forecast target (if forecast)
          index 2: forecast mask (if forecast)
          index 3: mort label
          index 4: los label
          index 5: sepsis label
          index 6: aki label
          index 7: kf label
        """
        if self.task == "forecast":
            targets = batch[1]
        elif self.task == "mort":
            targets = batch[3].float()
        elif self.task == "los":
            targets = batch[4]
        elif self.task == "sepsis":
            targets = batch[5].float()
        elif self.task in ["aki", "AKI"]:
            targets = batch[6].float()
        elif self.task == "kf":
            targets = batch[7]
        else:
            raise ValueError("Unsupported task in target selection")
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        return targets

    def evaluate_model(self, model, test_loader):
        model.to(self.device)
        model.eval()

        if self.task in ["forecast", "los", "kf"]:
            mae_metric = torchmetrics.MeanAbsoluteError().to(self.device)
            mse_metric = torchmetrics.MeanSquaredError().to(self.device)
            if self.task == "forecast":
                total_masked_abs_error = 0.0
                total_masked_sq_error = 0.0
                total_mask_count = 0.0
        elif self.task in ["mort", "sepsis", "aki", "AKI"]:
            accuracy_metric = torchmetrics.Accuracy(task="binary", threshold=0.5).to(
                self.device
            )
            auroc_metric = torchmetrics.AUROC(task="binary").to(self.device)
            auprc_metric = torchmetrics.AveragePrecision(task="binary").to(self.device)
        else:
            raise ValueError("Unsupported task for evaluation.")

        with torch.no_grad():
            for batch in test_loader:
                emb = batch[0].to(self.device).float()

                if getattr(self, "decoder", None) is not None:
                    emb_flat = emb.view(emb.size(0), -1)   # now [B, E]
                    # Now decode to [B, F_dec]:
                    dec_vec = self.decoder(emb_flat)       # [B, F_dec]
                    # Replicate across S → [B, S, F_dec]
                    dec = dec_vec.unsqueeze(1).expand(-1, self.S, -1)
                else:
                    dec = emb

                outputs = model(dec)
                outputs = outputs.contiguous()
        
                targets = self.get_targets(batch).to(self.device)
                targets = targets.contiguous()

                if self.task in ["forecast", "los", "kf"]:
                    mae_metric.update(outputs, targets)
                    mse_metric.update(outputs, targets)
                    if self.task == "forecast":
                        mask = batch[2].to(self.device).contiguous()
                        masked_diff = torch.abs(outputs - targets) * mask
                        masked_sq_diff = ((outputs - targets) ** 2) * mask
                        total_masked_abs_error += masked_diff.sum().item()
                        total_masked_sq_error += masked_sq_diff.sum().item()
                        total_mask_count += mask.sum().item()
                else:
                    probs = torch.sigmoid(outputs).contiguous()
                    accuracy_metric.update(probs, targets.int())
                    auroc_metric.update(probs, targets.int())
                    auprc_metric.update(probs, targets.int())

        if self.task in ["forecast", "los", "kf"]:
            mae = round(mae_metric.compute().item(), 5)
            mse = round(mse_metric.compute().item(), 5)
            result = {"mae": mae, "mse": mse}
            if self.task == "forecast":
                if total_mask_count > 0:
                    masked_mae = round(total_masked_abs_error / total_mask_count, 5)
                    masked_mse = round(total_masked_sq_error / total_mask_count, 5)
                else:
                    masked_mae, masked_mse = 0.0, 0.0
                result.update({"masked_mae": masked_mae, "masked_mse": masked_mse})
            return result
        else:
            accuracy = round(accuracy_metric.compute().item(), 5)
            auroc = round(auroc_metric.compute().item(), 5)
            auprc = round(auprc_metric.compute().item(), 5)
            return {"accuracy": accuracy, "auroc": auroc, "auprc": auprc}

    def test_all_models(self):
        """
        Iterate over all model checkpoint files in the runs directory, load and evaluate the models
        on the test set, and save the results (including metrics) to a CSV file.
        """
        results = []
        for filename in os.listdir(self.runs_dir):
            if not filename.endswith(".pt"):
                continue
            info = self.parse_model_filename(filename)
            if info is None:
                continue

            try:
                # Use the shared initialize_model with the parsed info.
                model = initialize_model(
                    info["model"],
                    self.task,
                    self.dataset,
                    info["representation"],
                    info["llm_type"],
                )
                model_path = os.path.join(self.runs_dir, filename)
                state_dict = torch.load(model_path, map_location="cpu")
                model.load_state_dict(state_dict)
                print(f"Loaded model from {filename} for evaluation.")
            except Exception as e:
                print(f"Failed to load or evaluate model from {filename}: {e}")
                continue

            # Build test dataset/loader based on whether the checkpoint is LLM-based.
            test_dataset = self.build_test_dataset(
                info["representation"], info["llm_type"]
            )
            test_loader = DataLoader(
                test_dataset, batch_size=self.args.batch_size, shuffle=False
            )

            metrics = self.evaluate_model(model, test_loader)
            print(f"Metrics for {filename}: {metrics}")

            # Add dataset, task, and also include llm_type (None for normal representations).
            info["dataset"] = self.dataset
            info["task"] = self.task
            info.update(metrics)
            results.append(info)

        # Define CSV file path.
        csv_filename = f"{self.dataset}_{self.task}_result.csv"
        csv_file = os.path.join(self.runs_dir, csv_filename)
        # Define CSV header order.
        header = [
            "dataset",
            "task",
            "model",
            "representation",
            "llm_type",
            "train_mask",
            "lr",
            "bs",
            "seed",
        ]
        if self.task == "forecast":
            header += ["mae", "mse", "masked_mae", "masked_mse"]
        elif self.task in ["los", "kf"]:
            header += ["mae", "mse"]
        else:
            header += ["accuracy", "auroc", "auprc"]

        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"Results saved to {csv_file}")
