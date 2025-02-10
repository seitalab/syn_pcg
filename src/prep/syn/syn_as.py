
from argparse import Namespace

import yaml
import numpy as np

from syn_pcg import PCGSynthesizer
from syn_utils import *
from syn_ar import low_pass_filter


cfg_file = "../../../config.yaml"
with open(cfg_file, "r") as f:
    cfg = yaml.safe_load(f)
EPS = 1e-10

def gen_as_noise(
    target_freq: int, 
    signal_duration: float, 
    noise_width: float, 
    start_loc_ratio: float, 
    end_loc_ratio: float,
    neg_ratio: float,
    min_wn_amp: float,
    max_wn_amp: float,
    as_noise_ma_width: float,
    low_pass_thres: float = None
):
    """
    Args:
        target_freq (int): Frequency of synthesized signal (step / sec).
        signal_duration (float): Duration of synthesized signal (sec).
        noise_width (float): Used as a sdev for bell curve mutiplied to white noise.
        start_loc_ratio (float): 
        end_loc_ratio (float): 
        neg_ratio (float):
        min_wn_amp (float):
        max_wn_amp (float):
        as_noise_ma_width (float):
        low_pass_thres (float): Cutoff frequency for low pass filter.
    Returns:
        noise (np.array): Noise signal
    """
    # Noise
    noise_duration =\
        signal_duration * (end_loc_ratio - start_loc_ratio)
    t = np.linspace(
        0, 
        signal_duration, 
        int(target_freq*noise_duration), 
        endpoint=False
    )
    max_wn_amp = max(max_wn_amp, min_wn_amp+EPS)
    wn = np.random.randn(len(t)) * (max_wn_amp - min_wn_amp)

    # Bell curve
    bell = asymmetric_bell_curve(
        len(t), noise_width, neg_ratio)

    # Combine
    noise_seg = wn * (bell + min_wn_amp)
    if as_noise_ma_width is not None:
        noise_seg = ma(noise_seg, int(as_noise_ma_width))
    
    start_loc = int(
        target_freq * signal_duration * start_loc_ratio)
    noise = np.zeros(int(target_freq * signal_duration))
    noise[start_loc:start_loc+len(noise_seg)] += noise_seg    
    
    if low_pass_thres is not None:
        noise = low_pass_filter(noise, low_pass_thres)

    return noise

class ASPCGSynthesizer(PCGSynthesizer):

    syn_type = "as_pcg"


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

        # Find the end of the S1 wave and add AS noise.
        i_wave_end = get_start_or_end_point(
            i_wave, 
            beat_params.i_end_threshold,
            get_start=False
        )
        as_noise = gen_as_noise(
            self.target_freq, 
            beat_params.beat_duration, 
            beat_params.as_noise_width,
            i_wave_end,
            beat_params.peak_ii_shift,
            beat_params.as_neg_ratio,
            beat_params.min_as_amp,
            beat_params.max_as_amp,
            beat_params.as_noise_ma_width,
            # beat_params.low_pass_thres
        )
        pcg += as_noise
    

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
        syn = ASPCGSynthesizer(syn_id, seed=seed)
        syn.make_dataset("train")
        syn.make_dataset("val")
        # break
    print("Done")
    
