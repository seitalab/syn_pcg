import os
import signal
import pickle
from argparse import Namespace

import yaml
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm

from syn_utils import *

cfg_file = "../../../config.yaml"
with open(cfg_file, "r") as f:
    cfg = yaml.safe_load(f)

EPS = 1e-10

def handle_timeout(signum, frame):
    print("Too long -> reset processing")
    raise TimeoutError("Overtime")

class PCGSynthesizer:
    
    syn_type = "normal_pcg"

    def __init__(self, syn_id: int, seed: int):
        np.random.seed(seed)

        self.seed = seed
        self.target_freq = cfg["synthesize"]["common"]["target_freq"]

        self._prep_save_loc(syn_id)
        self._set_cfg(syn_id)

    def _prep_save_loc(self, syn_id):

        self.save_loc = os.path.join(
            cfg["path"]["processed_data"], 
            cfg["synthesize"]["common"]["syndata_root"],
            self.syn_type,
            f"syn{syn_id:02d}",
            f"seed{self.seed:04d}"            
        )
        os.makedirs(self.save_loc, exist_ok=True)

    def _set_cfg(self, syn_id):
        syncfg_file = os.path.join(
            cfg["synthesize"]["common"]["syncfg_root"],
            self.syn_type,
            f'syn{syn_id:02d}.yaml'
        )

        with open(syncfg_file, "r") as f:
            syncfg = yaml.safe_load(f)
        self.syn_cfg = syncfg["syn_params"]
        self._save_cfg()

        # Other settings.
        self.target_length =\
            self.target_freq * syncfg["settings"]["duration"]
        self.sample_per_pkl = syncfg["settings"]["sample_per_pkl"]
        self.n_wavs_to_save = syncfg["settings"]["n_wavs_to_save"]
        self.n_syn = syncfg["settings"]["n_syn"]

    def generate_beat(self, beat_params: Namespace):
        """
        Args:
            beat_params (Namespace): 
        Returns:
            pseudo_ecg (np.ndarray): 
        """
        pcg = np.zeros(
            int(self.target_freq * beat_params.beat_duration))

        # S1
        i_wave = get_asymmetric_peak(
            self.target_freq, 
            beat_params.beat_duration, 
            beat_params.peak_i_fs, 
            beat_params.peak_i_height, 
            beat_params.peak_i_duration, 
            beat_params.peak_i_neg_ratio
        )
        pcg = concat_with_shift(
            pcg, i_wave, beat_params.peak_i_shift)

        # S2
        ii_wave = get_asymmetric_peak(
            self.target_freq, 
            beat_params.beat_duration, 
            beat_params.peak_ii_fs, 
            beat_params.peak_ii_height, 
            beat_params.peak_ii_duration, 
            beat_params.peak_ii_neg_ratio
        )
        pcg = concat_with_shift(
            pcg, ii_wave, beat_params.peak_ii_shift)

        # Base perturbations.
        base_i = base_sine(
            self.target_freq, 
            beat_params.beat_duration, 
            beat_params.base_i_amp, 
            beat_params.base_i_freq
        )
        base_ii = base_sine(
            self.target_freq, 
            beat_params.beat_duration, 
            beat_params.base_ii_amp, 
            beat_params.base_ii_freq
        )
        base_iii = base_sine(
            self.target_freq, 
            beat_params.beat_duration, 
            beat_params.base_iii_amp, 
            beat_params.base_iii_freq
        )
        pcg = concat_with_shift(pcg, base_i, 0.0)
        pcg = concat_with_shift(pcg, base_ii, 0.0)
        pcg = concat_with_shift(pcg, base_iii, 0.0)

        # Noise
        wn1 = white_noise(
            pcg.shape[0],
            int(beat_params.wn1_width), 
            beat_params.wn1_scaler
        )
        wn2 = white_noise(
            pcg.shape[0],
            int(beat_params.wn2_width), 
            beat_params.wn2_scaler
        )

        pseudo_pcg = pcg + wn1 + wn2
        return pseudo_pcg   

    def set_base_param(self):
        """
        Args:
            None
        Returns:
            base_param (Namespace): 
        """
        base_param = {}
        for key in self.syn_cfg:
            param_val = self.syn_cfg[key]["base"]["val"]
            if self.syn_cfg[key]["base"]["shift"] is not None:
                param_val += base_param[self.syn_cfg[key]["base"]["shift"]]

            # Add noise.
            noise_info = self.syn_cfg[key]["base_perturb"]
            if noise_info["type"] == "normal":
                param_val += np.random.normal(scale=noise_info["sdev"])
            elif noise_info["type"] == "uniform":
                rand_val = np.random.random()
                scale = noise_info["max"] - noise_info["min"]
                rand_val = rand_val * scale + noise_info["min"]
                param_val += rand_val
            base_param[key] = param_val
        return Namespace(**base_param)
    
    def perturb_param(self, base_params: Namespace):
        """
        Args:
            base_param (Namespace): 
        Returns:
            beat_param (Namespace): 
        """
        perturbed_param = {}
        for key, value in vars(base_params).items():
            # Add noise.
            noise_info = self.syn_cfg[key]["beat_perturb"]
            if noise_info["type"] == "normal":
                value += np.random.normal(scale=noise_info["sdev"])         
            elif noise_info["type"] == "uniform":
                rand_val = np.random.random()
                scale = noise_info["max"] - noise_info["min"]
                rand_val = rand_val * scale + noise_info["min"]
                value += rand_val
            perturbed_param[key] = value
        return Namespace(**perturbed_param)

    def generate_pcg(self):
        """
        Args:
        
        Returns:
        
        """
        p_pcg = np.array([0])
        base_params = self.set_base_param()
        while True:
            beat_params = self.perturb_param(base_params)
            _p_pcg = self.generate_beat(beat_params)
            
            p_pcg = np.concatenate([p_pcg, _p_pcg])

            if len(p_pcg) > self.target_length*1.5:
                break
        
        # Scale data.
        p_pcg = (p_pcg - p_pcg.min()) / (p_pcg.max() - p_pcg.min())
        p_pcg = (p_pcg - 0.5) * 2
        # p_pcg *= beat_params.max_scale

        # Randomly pick location.
        p_pcg = ma(p_pcg, int(base_params.ma_width))
        start_loc = np.random.choice(len(p_pcg) - self.target_length)
        p_pcg = p_pcg[start_loc:start_loc+self.target_length]

        return p_pcg

    def _save_cfg(self):
        """
        Args:

        Returns:

        """
        savename = os.path.join(
            self.save_loc,
            "cfg.txt"
        )
        cfg = ""
        for k, v in self.syn_cfg.items():
            cfg += f"{k} : {v}\n"
        with open(savename, "w") as f:
            f.write(cfg.strip())

    def _save_data(self, idx, signal):
        """
        Args:

        Returns:

        """
        dir_num = idx // 1000
        savename = os.path.join(
            self.save_loc,
            f"id{dir_num:04d}",
            f"id{idx:08d}.pkl"
        )

        with open(savename, "wb") as fp:
            pickle.dump(signal, fp)
        return "/".join(savename.split("/")[-2:])

    def _save_filelist(self, filelist, datatype):
        """
        Args:

        Returns:

        """
        savename = os.path.join(
            self.save_loc,
            f"{datatype}_files.txt"
        )
        with open(savename, "w") as f:
            f.write("\n".join(filelist))

    def synthesize_pcg(self, idx):
        """
        Args:

        Returns:

        """
        try:
            signal.signal(signal.SIGALRM, handle_timeout)
            signal.alarm(cfg["synthesize"]["common"]["max_process_time"])
            pcg = self.generate_pcg()

        # except TimeoutError:
        except:
            pcg = None

        signal.alarm(0)
        return pcg


    def _make_pickle(self, idx, datatype):
        start = idx * self.sample_per_pkl

        syn_pcg = []
        for idx_ in range(start, start+self.sample_per_pkl):
            syn_pcg.append(self.synthesize_pcg(idx_))
        
        syn_pcg = list(filter(lambda x: x is not None, syn_pcg))

        savename = os.path.join(
            self.save_loc,
            f"{datatype}_idx{idx+1:04d}.pkl"
        )
        with open(savename, "wb") as fp:
            pickle.dump(syn_pcg, fp)

        # Dump wav files.
        if self.n_wavs_to_save > 0:
            # Randomly pick indices to save.
            save_idxs = np.random.choice(
                len(syn_pcg), 
                min(self.n_wavs_to_save, len(syn_pcg)), 
                replace=False
            )
            for save_idx in save_idxs:
                self._dump_data_for_check(
                    start + save_idx, 
                    syn_pcg[save_idx]
                )
                self.n_wavs_to_save -= 1

    def _dump_data_for_check(self, idx, signal):
        """
        Args:

        Returns:

        """
        savename = f"idx{idx:08d}.wav"
        save_dir = os.path.join(
            self.save_loc,
            "dump"
        )
        os.makedirs(save_dir, exist_ok=True)
        savename = os.path.join(save_dir, savename)

        # Save wav file using soundfile.
        sf.write(savename, signal, self.target_freq)

        # Save plot.
        plt.figure(figsize=(20, 5))
        plt.plot(signal)
        plt.savefig(savename.replace(".wav", ".png"))
        plt.close()
  
    def make_dataset(self, datatype):
        """
        Args:

        Returns:

        """
        n_batch = self.n_syn[datatype]
        for idx in tqdm(range(n_batch)):
            self._make_pickle(idx, datatype)


if __name__ == "__main__":
    import sys

    try:
        syn_id = int(sys.argv[1])
    except:
        syn_id = 1

    for seed in range(6):
        print(f"Working on {seed} ...")
        syn = PCGSynthesizer(syn_id, seed=seed)
        syn.make_dataset("train")
        syn.make_dataset("val")
        # break
    print("Done")
    