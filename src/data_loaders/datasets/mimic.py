import torch
from torch.utils.data import Dataset
import numpy as np
import os
import time


class MimicDataset(Dataset):
    def __init__(self, data_path, y_path, split="train", train_percentile_mask="all", seed=42, perm=False):
        """
        Args:
            split (str): One of 'train', 'val', 'test'
            train_perncetile_mask (str): For the 'train' split, must be one of "all", "half", "quarter"
            N (int): Total number of samples.
            seed (int): Seed for reproducibility.
        """
        super().__init__()
        # Set seed for reproducibility
        np.random.seed(seed)

        t1 = time.time()
        data = np.load(data_path)

        all_x = data["all_x"]
        all_x_mask = data["all_x_mask"]
        timeseries_y = data["ts_y"]
        timeseries_y_mask = data["ts_y_mask"]
        # all_x = data["all_x_aligned"]
        # all_x_mask = data["all_x_mask_aligned"]
        # timeseries_y = data["ts_y_aligned"]
        # timeseries_y_mask = data["ts_y_mask_aligned"]

        # load y from y_path
        data = np.load(y_path)
        mortality_y = data["mort_y"]
        etimes_y = data["los_y"]
        time_to_discharge_y = data["dis_time_y"]
        # all lab features - 0/1 for each feature in the future 24 hours
        lab_labels_y = data["lab_labels"]
        # Vasopressors/Antibiotics - 0/1 for each feature in the future 24 hours
        feature_labels_y = data["feature_labels"]
        # ethics label
        gender_y = (data["gender_y"] == "M").astype(np.int64)
        age_y = data["age_y"]

        threshold_20_mask = data["th_20_mask"]
        threshold_50_mask = data["th_50_mask"]
        threshold_100_mask = data["th_100_mask"]
        train_mask = data["train_mask"]
        validation_mask = data["validation_mask"]
        test_mask = data["test_mask"]

        split_mask = {"train": train_mask, "val": validation_mask, "test": test_mask}[
            split
        ]
        assert sum(train_mask & validation_mask & test_mask) == 0

        threshold_mask = {
            "quarter": threshold_20_mask,
            "half": threshold_50_mask,
            "all": threshold_100_mask,
        }["all"]

        # -------- Compute train-only stats for normalization --------
        # Use the same train + percentile mask that filters training rows
        train_stats_mask = (train_mask & threshold_mask)

        _age_train = age_y[train_stats_mask].astype(np.float64)
        _et_train  = time_to_discharge_y[train_stats_mask].astype(np.float64)

        # NaN-safe mean/std; fallback std to 1.0 to avoid division by zero
        self.age_mean = float(np.nanmean(_age_train)) if _age_train.size else 0.0
        self.age_std  = float(np.nanstd(_age_train))  if _age_train.size else 1.0
        if self.age_std < 1e-8:
            self.age_std = 1.0

        self.et_mean = float(np.nanmean(_et_train)) if _et_train.size else 0.0
        self.et_std  = float(np.nanstd(_et_train))  if _et_train.size else 1.0
        if self.et_std < 1e-8:
            self.et_std = 1.0

        if split == "train":
            combined_mask = (split_mask & threshold_mask).astype(bool)
        else:
            combined_mask = split_mask.astype(bool)

        masked_x = all_x[combined_mask]
        masked_xm = all_x_mask[combined_mask]
        masked_y = timeseries_y[combined_mask]
        masked_ym = timeseries_y_mask[combined_mask]
        masked_mor = mortality_y[combined_mask]
        masked_etimes = time_to_discharge_y[combined_mask]
        masked_lab = lab_labels_y[combined_mask]
        masked_feature = feature_labels_y[combined_mask]
        masked_gender = gender_y[combined_mask]
        masked_age = age_y[combined_mask]

        self.N, self.T, self.D = masked_x.shape
        _, _, self.T2 = masked_y.shape

        # -------- Normalize age & time-to-discharge using train-only stats --------
        masked_age    = (masked_age - self.age_mean) / self.age_std
        masked_etimes = (masked_etimes - self.et_mean) / self.et_std
        
        # Create dummy data for each sample.
        # X: D x T matrix (input time series)
        self.X = masked_x
        # y1: D x T2 matrix (future time series)
        self.y1 = masked_y
        # m1: D x T2 binary mask for y1 (1 indicates the value is not missed)
        self.m1 = masked_ym
        # y2: binary label
        self.y2 = masked_mor
        # y3: regression label between 0 and 1
        self.y3 = masked_etimes
        # y4: lab labels: vector of 0/1
        self.y4 = masked_lab
        # y5: feature labels: vaso/antibiotics 0/1
        self.y5 = masked_feature
        # y6: gender
        self.y6 = masked_gender
        # y7: age
        self.y7 = masked_age

        if perm:
            rng = np.random.default_rng(2025)
            self.D_perm = rng.permutation(self.D)
            self.y1 = self.y1[:, :, self.D_perm]
            self.m1 = self.m1[:, :, self.D_perm]

        dt = time.time() - t1
        print(f"Dataset load time {dt}")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        real_idx = idx
        # Return a tuple: (X, y1, m1, y2, y3, y4, y5, y6, y7)
        return (
            torch.tensor(self.X[real_idx]),  # .transpose(0, 1),
            torch.tensor(self.y1[real_idx]),  # .transpose(0, 1),
            torch.tensor(self.m1[real_idx]),  # .transpose(0, 1),
            torch.tensor(self.y2[real_idx], dtype=torch.long),
            torch.tensor(self.y3[real_idx], dtype=torch.float32),
            torch.tensor(self.y4[real_idx], dtype=torch.int64),
            torch.tensor(self.y5[real_idx], dtype=torch.int64),
            torch.tensor(self.y6[real_idx], dtype=torch.int64),
            torch.tensor(self.y7[real_idx], dtype=torch.float32)
        )


class MimicLLMDataset(Dataset):
    def __init__(
        self,
        rep_datapath,
        data_path,
        split="train",
        llm_type="gemma",
        representation="sft",
        train_percentile_mask="all",
        seed=42,
        perm=False
    ):
        """
        Args:
            split (str): One of 'train', 'val', 'test'
            train_perncetile_mask (str): For the 'train' split, must be one of "all", "half", "quarter"
            N (int): Total number of samples.
            seed (int): Seed for reproducibility.
        """
        super().__init__()
        # Set seed for reproducibility
        np.random.seed(seed)

        t1 = time.time()
        assert representation in ["all", "sbf", "sbt", "TFM", "TSDE"]
        self.rep = representation
        assert os.path.exists(rep_datapath)
        assert os.path.exists(data_path)
        xdata = np.load(rep_datapath)
        data = np.load(data_path)
        assert llm_type in rep_datapath
        assert representation in rep_datapath

        # all_x = data["all_x"]
        all_x = xdata["embeddings"]
        print(all_x.shape)
        print(all_x[0].shape)
        if representation in ["all", "TFM", "TSDE"]:
            all_x = np.repeat(all_x[:, np.newaxis, :], repeats=1, axis=1)
        dt = time.time() - t1
        print(f"Embedding load time {dt}")

        all_x_mask = data["all_x_mask"]
        timeseries_y = data["ts_y"]
        timeseries_y_mask = data["ts_y_mask"]
        mortality_y = data["mort_y"]
        etimes_y = data["los_y"]
        time_to_discharge_y = data["dis_time_y"]
        # all lab features - 0/1 for each feature in the future 24 hours
        lab_labels_y = data["lab_labels"]
        # Vasopressors/Antibiotics - 0/1 for each feature in the future 24 hours
        feature_labels_y = data["feature_labels"]
        # ethics label
        gender_y = (data["gender_y"] == "M").astype(np.int64)
        age_y = data["age_y"]

        threshold_20_mask = data["th_20_mask"]
        threshold_50_mask = data["th_50_mask"]
        threshold_100_mask = data["th_100_mask"]
        train_mask = data["train_mask"]
        validation_mask = data["validation_mask"]
        test_mask = data["test_mask"]

        assert len(all_x_mask) == len(all_x)

        split_mask = {"train": train_mask, "val": validation_mask, "test": test_mask}[
            split
        ]

        threshold_mask = {
            "quarter": threshold_20_mask,
            "half": threshold_50_mask,
            "all": threshold_100_mask,
        }[train_percentile_mask]

        # -------- Compute train-only stats for normalization --------
        # Use the same train + percentile mask that filters training rows
        train_stats_mask = (train_mask & threshold_mask)

        _age_train = age_y[train_stats_mask].astype(np.float64)
        _et_train  = time_to_discharge_y[train_stats_mask].astype(np.float64)

        # NaN-safe mean/std; fallback std to 1.0 to avoid division by zero
        self.age_mean = float(np.nanmean(_age_train)) if _age_train.size else 0.0
        self.age_std  = float(np.nanstd(_age_train))  if _age_train.size else 1.0
        if self.age_std < 1e-8:
            self.age_std = 1.0

        self.et_mean = float(np.nanmean(_et_train)) if _et_train.size else 0.0
        self.et_std  = float(np.nanstd(_et_train))  if _et_train.size else 1.0
        if self.et_std < 1e-8:
            self.et_std = 1.0

        if split == "train":
            combined_mask = (split_mask & threshold_mask).astype(bool)
        else:
            combined_mask = split_mask.astype(bool)

        masked_x = all_x[combined_mask]
        masked_xm = all_x_mask[combined_mask]
        masked_y = timeseries_y[combined_mask]
        masked_ym = timeseries_y_mask[combined_mask]
        masked_mor = mortality_y[combined_mask]
        masked_etimes = time_to_discharge_y[combined_mask]
        masked_lab = lab_labels_y[combined_mask]
        masked_feature = feature_labels_y[combined_mask]
        masked_gender = gender_y[combined_mask]
        masked_age = age_y[combined_mask]

        self.N, self.T, self.D = masked_x.shape
        _, _, self.T2 = masked_y.shape

        # -------- Normalize age & time-to-discharge using train-only stats --------
        masked_age    = (masked_age - self.age_mean) / self.age_std
        masked_etimes = (masked_etimes - self.et_mean) / self.et_std

        # Create dummy data for each sample.
        # X: D x T matrix (input time series)
        self.X = masked_x
        # y1: D x T2 matrix (future time series)
        self.y1 = masked_y
        # m1: D x T2 binary mask for y1 (1 indicates the value is not missed)
        self.m1 = masked_ym
        # y2: binary label
        self.y2 = masked_mor
        # y3: regression label between 0 and 1
        self.y3 = masked_etimes
        # y4: binary label
        self.y4 = masked_lab
        # y5: feature labels: vaso/antibiotics 0/1
        self.y5 = masked_feature
        # y6: gender
        self.y6 = masked_gender
        # y7: age
        self.y7 = masked_age

        if perm:
            rng = np.random.default_rng(2025)
            self.D_perm = rng.permutation(self.D)
            self.X = self.X[:, :, self.D_perm]
            self.y1 = self.y1[:, :, self.D_perm]
            self.m1 = self.m1[:, :, self.D_perm]
            
        dt = time.time() - t1
        print(f"Dataset load time {dt}")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        real_idx = idx
        # Return a tuple: (X, y1, m1, y2, y3, y4, y5, y6)
        if self.rep == "sbt":
            x = torch.tensor(self.X[real_idx])
        elif self.rep == "sbf":
            x = torch.tensor(self.X[real_idx]).transpose(0, 1)
        elif self.rep == "all":
            x = torch.tensor(self.X[real_idx])
        else:
            x = torch.tensor(self.X[real_idx])
        return (
            x,  # .transpose(0, 1),
            torch.tensor(self.y1[real_idx]),  # .transpose(0, 1),
            torch.tensor(self.m1[real_idx]),  # .transpose(0, 1),
            torch.tensor(self.y2[real_idx], dtype=torch.long),
            torch.tensor(self.y3[real_idx], dtype=torch.float32),
            torch.tensor(self.y4[real_idx], dtype=torch.int64),
            torch.tensor(self.y5[real_idx], dtype=torch.int64),
            torch.tensor(self.y6[real_idx], dtype=torch.int64),
            torch.tensor(self.y7[real_idx], dtype=torch.float32)
        )


class MimicTSDEDataset(Dataset):
    """
    Loads TSDE embeddings saved as (N, D, T) (e.g., (N, 2112, 48)),
    converts to canonical (N, T, D) for consistency with the rest of the pipeline,
    applies the same split/percentile filtering and train-only normalization used before.
    """
    def __init__(
        self,
        rep_datapath: str,
        data_path: str,
        split: str = "train",                   # one of {"train","val","test"}
        train_percentile_mask: str = "all",     # {"all","half","quarter"}
        seed: int = 42,
        perm=False,
    ):
        super().__init__()
        assert split in {"train", "val", "test"}
        assert train_percentile_mask in {"all", "half", "quarter"}
        assert os.path.exists(rep_datapath), f"Missing rep_datapath: {rep_datapath}"
        assert os.path.exists(data_path),    f"Missing data_path: {data_path}"

        np.random.seed(seed)
        t0 = time.time()

        # -------- Load TSDE embeddings (expected (N, D, T)) --------
        xdata = np.load(rep_datapath)
        raw_x = xdata["embeddings"]

        if raw_x.ndim != 3:
            raise ValueError(f"Expected embeddings of shape (N, D, T); got {raw_x.shape}")

        all_x = np.transpose(raw_x, (0, 2, 1)).astype(np.float32)

        # -------- Load labels/masks from the structured .npz --------
        data = np.load(data_path)

        all_x_mask         = data["all_x_mask"]         # (N, T, D) mask for inputs
        timeseries_y       = data["ts_y"]               # (N, D_out, T2)
        timeseries_y_mask  = data["ts_y_mask"]          # (N, D_out, T2)
        mortality_y        = data["mort_y"]             # (N,)
        etimes_y           = data["los_y"]              # (N,)
        time_to_discharge_y= data["dis_time_y"]         # (N,)
        lab_labels_y       = data["lab_labels"]         # (N, n_lab)
        feature_labels_y   = data["feature_labels"]     # (N, n_feat)
        gender_y           = (data["gender_y"] == "M").astype(np.int64)  # (N,)
        age_y              = data["age_y"]              # (N,)

        threshold_20_mask  = data["th_20_mask"]
        threshold_50_mask  = data["th_50_mask"]
        threshold_100_mask = data["th_100_mask"]
        train_mask         = data["train_mask"]
        validation_mask    = data["validation_mask"]
        test_mask          = data["test_mask"]

        if len(all_x) != len(all_x_mask):
            raise ValueError(f"Input/mask length mismatch: {len(all_x)} vs {len(all_x_mask)}")

        # -------- Choose split & percentile mask --------
        split_mask = {"train": train_mask, "val": validation_mask, "test": test_mask}[split]
        pct_mask   = {"quarter": threshold_20_mask, "half": threshold_50_mask, "all": threshold_100_mask}[train_percentile_mask]

        # -------- Train-only stats for normalization --------
        # Use the *same* mask that filters training rows (train ∧ percentile)
        train_stats_mask = (train_mask & pct_mask).astype(bool)

        _age_train = age_y[train_stats_mask].astype(np.float64)
        _et_train  = time_to_discharge_y[train_stats_mask].astype(np.float64)  # or etimes_y

        self.age_mean = float(np.nanmean(_age_train)) if _age_train.size else 0.0
        self.age_std  = float(np.nanstd(_age_train))  if _age_train.size else 1.0
        if self.age_std < 1e-8: self.age_std = 1.0

        self.et_mean  = float(np.nanmean(_et_train)) if _et_train.size else 0.0
        self.et_std   = float(np.nanstd(_et_train))  if _et_train.size else 1.0
        if self.et_std < 1e-8: self.et_std = 1.0

        # -------- Apply filtering for the active split --------
        if split == "train":
            combined_mask = (split_mask & pct_mask).astype(bool)
        else:
            combined_mask = split_mask.astype(bool)

        masked_x    = all_x[combined_mask]            # (N', T, D)
        masked_xm   = all_x_mask[combined_mask]       # (N', T, D)
        masked_y    = timeseries_y[combined_mask]     # (N', D_out, T2)
        masked_ym   = timeseries_y_mask[combined_mask]
        masked_mor  = mortality_y[combined_mask]
        masked_et   = time_to_discharge_y[combined_mask]
        masked_lab  = lab_labels_y[combined_mask]
        masked_feat = feature_labels_y[combined_mask]
        masked_g    = gender_y[combined_mask]
        masked_age  = age_y[combined_mask]

        # Canonical shapes
        self.N, self.T, self.D = masked_x.shape
        _, self.D_out, self.T2 = masked_y.shape

        # -------- Normalize labels using train-only stats --------
        masked_age = (masked_age - self.age_mean) / self.age_std
        masked_et  = (masked_et  - self.et_mean)  / self.et_std

        # -------- Store --------
        self.X   = masked_x.astype(np.float32)    # (N', T, D)
        self.Xm  = masked_xm.astype(np.float32)   # keep if you need it later
        self.y1  = masked_y.astype(np.float32)    # (N', D_out, T2)
        self.m1  = masked_ym.astype(np.float32)   # (N', D_out, T2)
        self.y2  = masked_mor                     # mort (N',)
        self.y3  = masked_et.astype(np.float32)   # time-to-discharge, normalized
        self.y4  = masked_lab                     # lab labels
        self.y5  = masked_feat                    # feature labels
        self.y6  = masked_g                       # gender
        self.y7  = masked_age.astype(np.float32)  # age, normalized

        print(f"TSDE embeddings raw shape: {raw_x.shape} -> canonical X: {self.X.shape}")
        print(f"Dataset load time {time.time() - t0:.3f}s")

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # X is stored as (T, D). Return (T, D) for "sbt", or (D, T) for "sbf".
        x = self.X[idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(self.y1[idx], dtype=torch.float32),
            torch.tensor(self.m1[idx], dtype=torch.float32),
            torch.tensor(self.y2[idx], dtype=torch.long),
            torch.tensor(self.y3[idx], dtype=torch.float32),
            torch.tensor(self.y4[idx], dtype=torch.int64),
            torch.tensor(self.y5[idx], dtype=torch.int64),
            torch.tensor(self.y6[idx], dtype=torch.int64),
            torch.tensor(self.y7[idx], dtype=torch.float32),
        )
