import os
import pickle
from glob import glob
from typing import Optional, List

import yaml
import torch
import numpy as np
import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

cfg_file = "../../config.yaml"
with open(cfg_file, "r") as f:
    cfg = yaml.safe_load(f)

class BasePCGDataset(Dataset):

    target_dataset = None

    def __init__(
        self, 
        data_split: str,
        seed: int,
        target_dx: str,
        data_lim: Optional[int]=None, 
        transform: Optional[List]=None
    ) -> None:
        """
        Args:
            data_split (str): Dataset type to load (train, valid, test)
            seed (int): 
            data_lim (int): Total number of samples used for each class.
                If data_lim = 1000: pos = 1000, neg = 1000.
                In case, number of samples in pos/neg dataset is below data_lim / 2, 
                total number of samples used will be less than data_lim.
            transform (List): List of transformations to be applied.
        """
        assert(data_split in ["train", "val", "test"])
        
        # /export/work/users/nonaka/project/AMIpj/dataset_reid
        data_root = os.path.join(
            cfg["path"]["processed_data"],
            "dataset_reid",
            self.target_dataset
        )

        self.target_dx = target_dx
        self.data, self.labels, self.freqs = self._load_data(
            data_root, data_split, seed, data_lim)

        print(f"Loaded {data_split} set: {len(self.data)} samples.")

        self.transform = transform

    def _process_label(self, row: pd.Series) -> np.ndarray:
        """
        Process labels.
        Args:
            label (np.ndarray): Array of shape (num_samples,).
            dataset (str): Name of dataset.
        Returns:
            label (np.ndarray): Array of shape (num_samples,).
        """
        raise NotImplementedError
    
    def _data_to_array(self, data: np.ndarray) -> np.ndarray:

        return np.array(data, dtype="object")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        freq = self.freqs[index]

        if self.transform:
            sample = {"data": data, "src_freq": freq}
            sample = self.transform(sample)
            data = sample["data"]
        return data, torch.tensor(self.labels[index])

    def _load_data(
        self, 
        data_loc: str,
        data_split: str, 
        seed: int, 
        data_lim: Optional[int]
    ) -> np.ndarray:
        """
        Load file of target datatype.
        Args:
            data_split (str)
        Returns:
            X (np.ndarray): Array of shape (num_samples, 12, sequence_length).
        """
        # Load info csv.
        df_info = pd.read_csv(
            os.path.join(data_loc, "file_link_table.csv")
        )

        # Fetch target rows.
        target_split_idx = f"split_{seed}"
        if target_split_idx in df_info.columns:
            split_info = df_info.loc[:, target_split_idx].values
            df_target = df_info.loc[split_info == data_split]
        else:
            # If split info is not available, randomly re-split train/val.
            split_info = df_info.loc[:, "split_0"].values
            df_target = df_info.loc[split_info != "test"]
            idxs = np.arange(len(df_target))
            train_idx, val_idx = train_test_split(idxs, test_size=0.2, random_state=seed)
            if data_split == "train":
                df_target = df_target.iloc[train_idx]
            else:
                df_target = df_target.iloc[val_idx]

        # Limit data (if necessary).
        if data_lim is not None:
            # Randomly select `data_lim` samples.
            df_target = df_target.sample(n=data_lim, random_state=12345)
        
        # Load data.
        data, labels, freqs = [], [], []
        for _, row in tqdm(df_target.iterrows(), total=len(df_target)):
            # Wav data.
            filename = row["rename"].replace("rename/", "") # Remove `rename/` prefix.
            data_path = os.path.join(data_loc, filename)
            pcg, freq = sf.read(data_path)

            # Label.
            label = self._process_label(row)

            # Append.
            data.append(pcg)
            labels.append(label)
            freqs.append(freq)

        data = self._data_to_array(data)
        labels = np.array(labels)
        freqs = np.array(freqs)
        data, labels, freqs = self._filter_data(data, labels, freqs)
        return data, labels, freqs
    
    def _filter_data(self, data, labels, freqs):
        """
        Filter data.
        Args:
            data (np.ndarray): Array of shape (num_samples,).
            labels (np.ndarray): Array of shape (num_samples,).
            freqs (np.ndarray): Array of shape (num_samples,).
        Returns:
            data (np.ndarray): Array of shape (num_samples,).
            labels (np.ndarray): Array of shape (num_samples,).
            freqs (np.ndarray): Array of shape (num_samples,).
        """
        return data, labels, freqs

class BUETDataset(BasePCGDataset):

    target_dataset = "buet"
    dx_list = ["AR", "AS", "MR", "MS", "Gender", "Smoker"]

    def _process_label(self, row):
        """
        Process labels.
        Args:
            label (np.ndarray): Array of shape (num_samples,).
            dataset (str): Name of dataset.
        Returns:
            label (np.ndarray): Array of shape (num_samples,).
        """
        assert self.target_dx in self.dx_list

        label = row[self.target_dx]
        if self.target_dx == "Gender":
            label = 0 if label == "M" else 1

        return label

class PrivatePCGDataset(BasePCGDataset):

    target_dataset = "private"
    dx_list = ["AR", "AS", "MR", "SEX"]
    exclude_hosp = ["kk", "os"]

    def _process_label(self, row):

        assert self.target_dx in self.dx_list

        label = row[f"{self.target_dx}_label"]
        
        if self.target_dx == "SEX":
            label = 0 if label == "M" else 1

        # Check if hospital is in exclude list.
        if row["hospital"] in self.exclude_hosp:
            label = -1
        
        # Filter if any -1 label.
        for dx in self.dx_list[:-1]:
            if row[f"{dx}_label"] == -1:
                label = -1

        return label

    def _filter_data(self, data, labels, freqs):
        """
        Filter data.
        Args:
            data (np.ndarray): Array of shape (num_samples,).
            labels (np.ndarray): Array of shape (num_samples,).
            freqs (np.ndarray): Array of shape (num_samples,).
        Returns:
            data (np.ndarray): Array of shape (num_samples,).
            labels (np.ndarray): Array of shape (num_samples,).
            freqs (np.ndarray): Array of shape (num_samples,).
        """
        # Filter out samples with label -1.
        mask = labels != -1
        data = data[mask]
        labels = labels[mask]
        freqs = freqs[mask]

        return data, labels, freqs

class SynPCGDataset(BasePCGDataset):

    target_dataset = "syn"
    dx_list = ["AR", "AS", "MR"]

    def __init__(
        self, 
        data_split: str,
        seed: int,
        target_dx: str,
        dataset_ver_norm: str,
        dataset_ver_dx: str,
        data_lim: Optional[int]=None, 
        transform: Optional[List]=None
    ) -> None:
        """
        Args:
            data_split (str): Dataset type to load (train, valid, test)
            seed (int): 
            dataset_ver (str): Dataset version. (v001, v002, ...)
            data_lim (int): Total number of samples used for each class.
            transform (List): List of transformations to be applied.
        """
        assert(data_split in ["train", "val", "test"])
        
        data_root = os.path.join(
            cfg["path"]["processed_data"],
            "dataset_syn",
        )

        self.dataset_ver_norm = dataset_ver_norm
        self.dataset_ver_dx = dataset_ver_dx
        self.target_dx = target_dx
        
        self.data, self.labels, self.freqs = self._load_data(
            data_root, data_split, seed, data_lim)

        print(f"Loaded {data_split} set: {len(self.data)} samples.")

        self.transform = transform

    def __load_pickle(
        self, 
        data_root: str, 
        dx_dir: str, 
        data_split: str, 
        seed: int, 
        data_lim: int
    ):
        """
        Args:
            data_root (str):
            dx_dir (str):
            data_split (str):
            seed (int):
            data_lim (int):
        Returns:
            pcgs (np.ndarray): Array of shape (num_samples,).
        """
        # Force to use seed 5 if seed is not in [0, 1, 2, 3, 4].
        if seed not in [0, 1, 2, 3, 4]:
            seed = 5

        if dx_dir == "normal_pcg":
            dataset_ver = self.dataset_ver_norm
        else:
            dataset_ver = self.dataset_ver_dx

        # Fetch target files.
        data_dir = os.path.join(
            data_root, 
            dx_dir, 
            dataset_ver, 
            f"seed{seed:04d}"
        )
        target_pkls = sorted(glob(data_dir + f"/{data_split}_idx*.pkl"))

        print(f"Loading from {dx_dir}, {len(target_pkls)} files ...")
        pcgs = []
        for pkl in tqdm(target_pkls):
            with open(pkl, "rb") as f:
                pcg = pickle.load(f)
            pcgs += pcg

            if data_lim is not None:
                if len(pcgs) >= data_lim:
                    break
        
        pcgs = np.array(pcgs)
        if data_lim is not None:
            pcgs = pcgs[:data_lim]
        return pcgs

    def _load_data(
        self, 
        data_root: str,
        data_split: str, 
        seed: int, 
        data_lim: Optional[int]
    ) -> np.ndarray:
        """
        Load file of target datatype.
        Args:
            data_split (str)
        Returns:
            X (np.ndarray): Array of shape (num_samples, 12, sequence_length).
        """
        if data_lim is not None:
            data_lim_per_class = data_lim // 2
        else:
            data_lim_per_class = None

        # Load negative data.
        neg_pcg = self.__load_pickle(
            data_root, "normal_pcg", data_split, seed, data_lim_per_class
        )
        neg_labels = np.zeros(len(neg_pcg))

        # Load positive data.
        dx_dir = self.target_dx.lower() + "_pcg"
        pos_pcg = self.__load_pickle(
            data_root, dx_dir, data_split, seed, data_lim_per_class
        )
        pos_labels = np.ones(len(pos_pcg))

        # Concat.
        data = np.concatenate([neg_pcg, pos_pcg])
        labels = np.concatenate([neg_labels, pos_labels])
        freqs = np.array([cfg["synthesize"]["common"]["target_freq"]] * len(data))
        
        return data, labels, freqs