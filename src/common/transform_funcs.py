import pickle

import numpy as np
import torch

np.random.seed(0)

class PickleOpener:

    def __call__(self, sample):

        with open(sample["data"], "rb") as fp:
            data = pickle.load(fp)

        sample.update(data=data)
        return sample

class AlignLength:

    def __init__(self, target_len: int):

        self.target_len = target_len

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (data_length, )}
        Returns:
            sample (Dict): {"data": Array of shape (data_length, )}
        """
        data = sample["data"]

        if len(data) < self.target_len:
            total_pad = self.target_len - len(data)
            pad_l = int(np.random.rand() * total_pad)
            pad_r = total_pad - pad_l
            data = np.concatenate([
                np.zeros(pad_l),
                data,
                np.zeros(pad_r)
            ])
        
        if len(data) > self.target_len:
            total_cut = len(data) - self.target_len
            cut_l = int(np.random.rand() * total_cut)
            data = data[cut_l:cut_l+self.target_len]

        sample.update(data=data)
        return sample

class ScaleSignal:

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (data_length, )}
        Returns:
            sample (Dict): {"data": Array of shape (data_length, )}
        """
        data = sample["data"]

        scaled_data = (data - data.mean()) / data.std()

        sample.update(data=scaled_data)
        return sample

class SplitIntoChunk:

    def __init__(self, target_len: int, overlap: float=0.2):

        self.target_len = target_len
        self.shift_len = int(target_len * (1 - overlap))

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (data_length, )}
        Returns:
            sample (Dict): {"data": Array of shape (data_length, )}
        """
        data = sample["data"]
        
        n_chunk = (len(data) - self.shift_len) // self.shift_len
        # print(n_chunk, len(data), self.shift_len)
        if n_chunk < 1:
            if len(data) < self.target_len:
                total_pad = self.target_len - len(data)
                pad_l = int(np.random.rand() * total_pad)
                pad_r = total_pad - pad_l
                chunks = np.concatenate([
                    np.zeros(pad_l),
                    data,
                    np.zeros(pad_r)
                ])
            
            elif len(data) > self.target_len:
                total_cut = len(data) - self.target_len
                cut_l = int(np.random.rand() * total_cut)
                chunks = data[cut_l:cut_l+self.target_len]
            
            else:
                chunks = data
            chunks = data[np.newaxis]
        else:
            n_chunk = (len(data) - self.shift_len) // self.shift_len
            start = np.arange(n_chunk) * self.shift_len
            chunks = np.array([data[s:s+self.target_len] for s in start])
        sample.update(data=chunks)
        return sample

class ExtractCenter:

    def __init__(self, target_len: int):

        self.target_len = target_len

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (data_length, )}
        Returns:
            sample (Dict): {"data": Array of shape (data_length, )}
        """
        data = sample["data"]

        if len(data) < self.target_len:
            total_pad = self.target_len - len(data)
            pad_l = int(np.random.rand() * total_pad)
            pad_r = total_pad - pad_l
            data = np.concatenate([
                np.zeros(pad_l),
                data,
                np.zeros(pad_r)
            ])
        elif len(data) == self.target_len:
            pass
        else:
            start = int((len(data) // 2) - (self.target_len // 2))
            data = data[start:start+self.target_len]

        sample.update(data=data)
        return sample

class Downsampler:

    def __init__(self, target_freq: int):

        self.target_freq = target_freq

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (data_length, )}
        Returns:
            sample (Dict): {"data": Array of shape (data_length, )}
        """
        data = sample["data"]
        src_freq = sample["src_freq"]
        assert src_freq >= self.target_freq
        ds = src_freq // self.target_freq

        # Cut off the end of the signal to make it evenly divisible by ds.
        if data.shape[0] % ds != 0:
            data = data[:-(data.shape[0] % ds)]

        if data.ndim == 1:
            downsampled_signal = data.reshape(-1, ds).mean(axis=1)
        else:
            n_chunks = data.shape[1]
            downsampled_signal = data.reshape(n_chunks, ds).mean(axis=-1)

        # if dtype = "object", then change to float32.
        if data.dtype == "object":
            downsampled_signal = downsampled_signal.astype(np.float32)
        
        sample.update(data=downsampled_signal)
        return sample

class ToTensor:
    """Convert ndarrays in sample to Tensors."""
    # def __init__(self, with_label: bool):

    #     self.with_label = with_label

    def __call__(self, sample):

        # if self.load_masked:
        #     data, masked_data = sample["data"], sample["masked"]
        #     data_tensor = torch.from_numpy(data)
        #     masked_tensor = torch.from_numpy(masked_data).unsqueeze(0)
        #     sample = {"data": data_tensor, "masked": masked_tensor}
        # else:
        try:
            data_tensor = torch.from_numpy(sample["data"])
        except:
            print(sample)
            aaa

        data_tensor = data_tensor.unsqueeze(-1)
        # if self.with_label:
        #     label_tensor = torch.Tensor([sample["label"]])
        # else:
        #     label_tensor = None
        sample.update(data=data_tensor)#, label=label_tensor)
        return sample
