from argparse import Namespace

import yaml
import numpy as np

from syn_ar import low_pass_filter
from syn_as import ASPCGSynthesizer
from syn_utils import *

cfg_file = "../../../config.yaml"
with open(cfg_file, "r") as f:
    cfg = yaml.safe_load(f)

EPS = 1e-10

def gen_mr_noise(
    target_freq: int, 
    noise_duration: float, 
    min_wn_amp: float,
    max_wn_amp: float,
    mr_noise_ma_width: float,
    low_pass_thres: float = None
):
    """
    Args:
        target_freq (int): Frequency of synthesized signal (step / sec).
        noise_duration (float): Duration of synthesized signal (sec).
        noise_width (float): Used as a sdev for bell curve mutiplied to white noise.
        neg_ratio (float):
        min_wn_amp (float):
        max_wn_amp (float):
    Returns:    
        noise (np.array): Noise signal
    """
    t = np.linspace(
        0, 
        noise_duration, 
        int(target_freq*noise_duration), 
        endpoint=False
    )
    max_wn_amp = max(max_wn_amp, min_wn_amp+EPS)

    noise = \
        np.random.randn(len(t)) * \
            (max_wn_amp - min_wn_amp) + min_wn_amp
    
    if mr_noise_ma_width is not None:
        noise = ma(noise, int(mr_noise_ma_width), mode="same")

    # Low pass filter
    if low_pass_thres is not None:
        noise = low_pass_filter(noise, low_pass_thres)

    return noise 

class MRPCGSynthesizer(ASPCGSynthesizer):
    
    syn_type = "mr_pcg"

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

        # Find the start of the S1 wave and add MR noise.
        i_wave_start = get_start_or_end_point(
            i_wave, 
            beat_params.i_start_threshold,
            get_start=True
        ) 
        i_wave_start = int((i_wave_start + + beat_params.peak_i_shift) * self.target_freq)
        ii_wave_end = get_start_or_end_point(
            ii_wave,
            beat_params.ii_end_threshold,
            get_start=False
        )
        ii_wave_end = int((ii_wave_end + beat_params.peak_ii_shift) * self.target_freq)

        # Add MR noise between S1 and S2.
        noise_duration_step = (ii_wave_end - i_wave_start) * beat_params.mr_noise_len_ratio
        noise_duration_time = noise_duration_step / self.target_freq
        mr_noise = gen_mr_noise(
            self.target_freq, 
            noise_duration_time,
            beat_params.min_mr_amp,
            beat_params.max_mr_amp,
            mr_noise_ma_width=beat_params.mr_noise_ma_width,
            # low_pass_thres=beat_params.mr_low_pass_thres
        )

        if len(mr_noise) != noise_duration_step:
            assert np.abs(len(mr_noise) - noise_duration_step) < 10 # Acceptable error
            if len(mr_noise) > noise_duration_step:
                pcg[i_wave_start:ii_wave_end] = mr_noise[:noise_duration_step]
            else:
                if len(mr_noise) > len(pcg)-i_wave_start:
                    pcg[i_wave_start:] = mr_noise[:len(pcg)-i_wave_start]
                else:
                    pcg[i_wave_start:i_wave_start+len(mr_noise)] += mr_noise
        else:
            pcg[i_wave_start:ii_wave_end] += mr_noise

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


if __name__ == "__main__":
    import sys

    try:
        syn_id = int(sys.argv[1])
    except:
        syn_id = 1

    for seed in range(6):
        print(f"Working on {seed} ...")
        syn = MRPCGSynthesizer(syn_id, seed=seed)
        syn.make_dataset("train")
        syn.make_dataset("val")
        # break
    print("Done")
    