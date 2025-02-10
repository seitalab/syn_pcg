import signal
from argparse import Namespace

import yaml
import numpy as np

from syn_pcg import PCGSynthesizer
from syn_utils import *

cfg_file = "../../../config.yaml"
with open(cfg_file, "r") as f:
    cfg = yaml.safe_load(f)

EPS = 1e-10

def low_pass_filter(signal, cutoff):
    """
    Args:
        signal (np.array): Signal to be filtered.
        cutoff (float): Cutoff frequency.
    Returns:
        filtered_signal (np.array): Filtered signal.
    """
    from scipy.signal import butter, lfilter

    nyquist = 0.5 * cfg["synthesize"]["common"]["target_freq"]
    normal_cutoff = cutoff / nyquist
    b, a = butter(1, normal_cutoff, btype='low', analog=False)
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal

def gen_ar_noise(
    target_freq: int, 
    noise_duration: float, 
    noise_width: float,
    neg_ratio: float,
    min_wn_amp: float,
    max_wn_amp: float,
    ar_noise_ma_width: float,
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
        low_pass_thres (float): Cutoff frequency for low pass filter.
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

    wn = np.random.randn(len(t)) * (max_wn_amp - min_wn_amp)
    bell = asymmetric_bell_curve(len(t), noise_width, neg_ratio)
    
    noise = wn * (bell + min_wn_amp)
    if ar_noise_ma_width is not None:
        noise = ma(noise, int(ar_noise_ma_width), mode="same")
    
    # Apply low pass filter.
    if low_pass_thres is not None:
        noise = low_pass_filter(noise, low_pass_thres)
    return noise 

class ARPCGSynthesizer(PCGSynthesizer):
    
    syn_type = "ar_pcg"

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

        i_wave_start = get_start_or_end_point(
            i_wave, 
            beat_params.i_end_threshold, 
            get_start=True, 
            return_idx=True
        )
        i_wave_start += \
            beat_params.peak_i_shift * \
            int(self.target_freq * beat_params.beat_duration)


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

        ii_wave_end = get_start_or_end_point(
            ii_wave, 
            beat_params.ii_start_threshold, 
            get_start=False, 
            return_idx=True
        )
        ii_wave_end += \
            beat_params.peak_ii_shift * \
            int(self.target_freq * beat_params.beat_duration)

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
        return pseudo_pcg, int(i_wave_start), int(ii_wave_end)

    def _add_noise_between_s1_and_s2(
        self, 
        p_pcg, 
        beat_params, 
        prev_ii_end,
        i_start_after_concat,
    ):
        """
        Args:
        
        Returns:

        """
        noise_duration =\
            (i_start_after_concat - prev_ii_end) / self.target_freq
        noise = gen_ar_noise(
            self.target_freq, 
            noise_duration, 
            beat_params.ar_noise_width,
            beat_params.ar_neg_ratio,
            beat_params.min_ar_amp,
            beat_params.max_ar_amp,
            beat_params.ar_noise_ma_width,
            # beat_params.low_pass_thres
        )
        noise_len = i_start_after_concat - prev_ii_end
        if len(noise) != noise_len:
            assert np.abs(len(noise) - noise_len) < 10 # Acceptable error
            if len(noise) > noise_len:
                p_pcg[prev_ii_end:i_start_after_concat] = noise[:noise_len]
            else:
                p_pcg[prev_ii_end:prev_ii_end+len(noise)] += noise
        else:
            p_pcg[prev_ii_end:i_start_after_concat] += noise
        return p_pcg

    def generate_pcg(self):
        """
        Args:
        
        Returns:
        
        """
        p_pcg = np.array([0])
        base_params = self.set_base_param()
        prev_ii_end = None

        while True:
            beat_params = self.perturb_param(base_params)
            _p_pcg, i_start, ii_end = self.generate_beat(beat_params)
            p_pcg = np.concatenate([p_pcg, _p_pcg])

            # Remove first element if first iteration.
            if len(p_pcg) == len(_p_pcg) + 1:
                p_pcg = p_pcg[1:]

            # Add AR noise (after S2 until next S1).
            i_start_after_concat = len(p_pcg) + i_start - len(_p_pcg)
            if prev_ii_end is not None:
                p_pcg = self._add_noise_between_s1_and_s2(
                    p_pcg, 
                    beat_params, 
                    prev_ii_end,
                    i_start_after_concat,
                )
            prev_ii_end = ii_end + len(p_pcg) - len(_p_pcg)

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
        signal.alarm(0)

        return p_pcg

if __name__ == "__main__":
    import sys

    try:
        syn_id = int(sys.argv[1])
    except:
        syn_id = 1

    for seed in range(6):
        print(f"Working on {seed} ...")
        syn = ARPCGSynthesizer(syn_id, seed=seed)
        syn.make_dataset("train")
        syn.make_dataset("val")
        # break
    print("Done")
    