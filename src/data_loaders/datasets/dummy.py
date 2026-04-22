import torch
from torch.utils.data import Dataset
import numpy as np


class DummyTSDataset(Dataset):
    def __init__(self, split="train", train_mask="all", seed=42):
        """
        Args:
            split (str): One of 'train', 'val', 'test'
            train_mask (str): For the 'train' split, must be one of "all", "half", "quarter"
            N (int): Total number of samples.
            seed (int): Seed for reproducibility.
        """
        super().__init__()
        # Set seed for reproducibility
        np.random.seed(seed)
        N = 10000  # Total number of samples
        self.N = 10000  # Total number of samples

        # In practice, these dimensions will be determined by the files you load.
        self.D = 61  # Number of features
        self.T = 48  # Number of timesteps in the input time series
        self.T2 = 24  # Number of timesteps in the future time series (y1)

        # Create dummy data for each sample.
        # X: D x T matrix (input time series)
        self.X = [np.random.randn(self.D, self.T).astype(np.float32) for _ in range(N)]
        # y1: D x T2 matrix (future time series)
        self.y1 = [
            np.random.randn(self.D, self.T2).astype(np.float32) for _ in range(N)
        ]
        # m1: D x T2 binary mask for y1 (1 indicates the value is not missed)
        self.m1 = [
            np.random.randint(0, 2, size=(self.D, self.T2)).astype(np.float32)
            for _ in range(N)
        ]
        # y2: binary label
        self.y2 = [np.random.randint(0, 2) for _ in range(N)]
        # y3: regression label between 0 and 1
        self.y3 = [np.random.rand() for _ in range(N)]
        # y4: binary label
        self.y4 = [np.random.randint(0, 2) for _ in range(N)]
        # y5: binary label
        self.y5 = [np.random.randint(0, 2) for _ in range(N)]
        # y6: regression label between 0 and 1
        self.y6 = [np.random.rand() for _ in range(N)]

        # Create indices and shuffle them for splitting
        indices = np.arange(N)
        np.random.shuffle(indices)
        n_train = int(0.7 * N)
        n_val = int(0.1 * N)
        n_test = N - n_train - n_val  # remainder goes to test

        # Split indices for train, val, test
        self.train_idx = indices[:n_train]
        self.val_idx = indices[n_train : n_train + n_val]
        self.test_idx = indices[n_train + n_val :]

        # Depending on the split, select the corresponding indices.
        if split == "train":
            # Ensure that train_mask is valid.
            if train_mask not in ["all", "half", "quarter"]:
                raise ValueError(
                    "train_mask must be one of 'all', 'half', or 'quarter'"
                )
            if train_mask == "all":
                self.selected_idx = self.train_idx
            else:
                train_indices = self.train_idx.copy()
                np.random.shuffle(train_indices)
                if train_mask == "half":
                    n_selected = int(0.5 * len(train_indices))
                elif train_mask == "quarter":
                    n_selected = int(0.25 * len(train_indices))
                self.selected_idx = train_indices[:n_selected]
        elif split == "val":
            self.selected_idx = self.val_idx
        elif split == "test":
            self.selected_idx = self.test_idx
        else:
            raise ValueError("Invalid split: choose among 'train', 'val', 'test'")

    def __len__(self):
        return len(self.selected_idx)

    def __getitem__(self, idx):
        real_idx = self.selected_idx[idx]
        # Return a tuple: (X, y1, m1, y2, y3, y4, y5, y6)
        return (
            torch.tensor(self.X[real_idx]).transpose(0, 1),
            torch.tensor(self.y1[real_idx]).transpose(0, 1),
            torch.tensor(self.m1[real_idx]).transpose(0, 1),
            torch.tensor(self.y2[real_idx], dtype=torch.long),
            torch.tensor(self.y3[real_idx], dtype=torch.float32),
            torch.tensor(self.y4[real_idx], dtype=torch.long),
            torch.tensor(self.y5[real_idx], dtype=torch.long),
            torch.tensor(self.y6[real_idx], dtype=torch.float32),
        )


if __name__ == "__main__":
    dataset = DummyTSDataset(split="train", train_mask="all")
    print("Dataset length:", len(dataset))
    print("Dataset sample shape:", dataset[0][0].shape)
    print("Dataset sample shape:", dataset[0][1].shape)
    print("Dataset sample shape:", dataset[0][2].shape)
    print("Dataset sample shape:", dataset[0][3])
    print("Dataset sample shape:", dataset[0][4])
    print("Dataset sample shape:", dataset[0][5])
    print("Dataset sample shape:", dataset[0][6])
    print("Dataset sample shape:", dataset[0][7])
