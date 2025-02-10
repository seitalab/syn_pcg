
import numpy as np

def shift_wave(arr: np.ndarray, shift_len: int, shift_right: bool):
    """
    Args:
        arr (np.ndarray): Input array
        shift_len (int): Number of elements to shift
        shift_right (bool): Shift direction
    Returns:
        shifted_arr (np.ndarray): Shifted array
    """
    if shift_len < 0:
        return arr
    
    if shift_right:
        shifted_arr = np.concatenate((np.zeros(shift_len), arr[:-shift_len]))
    else:
        shifted_arr = np.concatenate((arr[shift_len:], np.zeros(shift_len)))
    return shifted_arr

def random_shift(arr: np.ndarray, drop_ratio: float=0.05):
    """
    Randomly drop the array from the head or tail and add zeros to the other side.
    
    Parameters:
        arr (numpy.ndarray): The input array.
        drop_ratio (float): Ratio of the array to drop.
        
    Returns:
        numpy.ndarray: The modified array.
    """
    if drop_ratio <= 0 or drop_ratio > 1:
        raise ValueError("drop_percent must be between 0 and 100.")
    
    # Calculate the number of elements to drop
    drop_length = int(drop_ratio * len(arr))
    
    # Decide randomly whether to drop from the head or tail
    drop_from_tail = np.random.choice([True, False])
    modified_arr = shift_wave(arr, drop_length, drop_from_tail)
    return modified_arr

def asymmetric_bell_curve(gen_len: int, std_pos:float, neg_ratio: float):
    """
    Args:
        gen_len (int): Length of the generated curve
        std_pos (float): Standard deviation for positive side
        neg_ratio (float): Ratio of standard deviation for negative side to positive side
    Returns:
        asymmetric_bell (np.array): Asymmetric bell curve
    """
    std_neg = std_pos * neg_ratio  # Standard deviation for negative side

    # Create an array of x values
    x = np.linspace(-10, 10, gen_len)

    # Scaling factors to ensure same peak at x = 0
    scaling_pos = 1 / (std_pos * np.sqrt(2 * np.pi))  # Scaling factor for positive side
    scaling_neg = 1 / (std_neg * np.sqrt(2 * np.pi)) / (std_pos / std_neg)  # Scaling factor for negative side

    # Define the bell curve with different decay rates for positive and negative sides
    asymmetric_bell = np.where(x >= 0, 
                 scaling_pos * np.exp(-0.5 * (x / std_pos)**2), 
                 scaling_neg * np.exp(-0.5 * (x / std_neg)**2))
    return asymmetric_bell

def gen_sine_wave(target_freq: int, signal_duration: float, sine_freq: int):
    """
    Args:
        target_freq (int): step / sec
        signal_duration (float): sec
        sine_freq (int): 
    """
    t = np.linspace(
        0, signal_duration, int(target_freq*signal_duration), endpoint=False)
    sine_wave = np.sin(2 * np.pi * sine_freq * t)
    return sine_wave

def concat_with_shift(
    base_wave: np.ndarray, 
    add_wave, 
    shift_ratio
):
    if shift_ratio != 0:    
        shift_len = int(shift_ratio * len(add_wave))
        add_wave = shift_wave(add_wave, shift_len, shift_right=True)
    return base_wave + add_wave

def base_sine(
    target_freq: int, 
    signal_duration: float, 
    amplitude: float, 
    frequency: float, 
):
    """
    Args:
        target_freq (int): Frequency of synthesized signal (step / sec).
        signal_duration (float): Duration of synthesized signal (sec).
        amplitude (float): Amplitude of synthesized signal.
        frequency (float): Frequency of synthesized signal (Hz).
    Returns:
        base (np.array): Sine wave signal
    """
    t = np.linspace(
        0, 
        signal_duration, 
        int(target_freq*signal_duration), 
        endpoint=False
    )
    base = amplitude * np.sin(2 * np.pi * frequency * t)
    return base

def get_asymmetric_peak(
    target_freq: int,
    signal_duration: float, 
    peak_fs: float, 
    peak_height: float, 
    peak_duration: float, 
    neg_side_ratio: float, 
    start_point_val: float=0.0001
):
    """
    Args:
        target_freq (int): Frequency of synthesized signal (step / sec).
        signal_duration (float): Duration of synthesized signal (sec).
        peak_fs (float): 
        peak_height (float): 
        peak_duration (float): 
        neg_side_ratio (float): 
        start_point_val (float): 
    """
    sine_wave = gen_sine_wave(
        target_freq, signal_duration, peak_fs)
    y = asymmetric_bell_curve(
        int(target_freq*signal_duration), 
        peak_duration, 
        neg_side_ratio
    )
    wave = random_shift(y) * sine_wave
    
    shift_len = np.where(np.abs(wave) > start_point_val)[0][0]
    wave = shift_wave(wave, shift_len, shift_right=False)
    wave = wave / np.abs(wave).max() * peak_height
    return wave


def get_start_or_end_point(
    wave: np.ndarray, 
    threshold_val: float,
    get_start: bool,
    return_idx: bool=False
):
    """
    Args:
        wave (np.ndarray): 
        threshold_val (float):
        get_start (bool):
        return_idx (bool):
    Returns:
        start_or_end_point (float):
    """
    candidates = np.where(np.abs(wave) > threshold_val)[0]
    if get_start:
        start_or_end_idx = candidates[0]
    else:
        start_or_end_idx = candidates[-1]

    if return_idx:
        return start_or_end_idx

    start_or_end_point = start_or_end_idx / len(wave)
    return start_or_end_point

def ma(x, w, mode='valid'):
    return np.convolve(x, np.ones(w), mode) / w

def white_noise(
    wn_length, 
    noise_width, 
    scaler
):
    pad = int(wn_length -  int(wn_length / noise_width) * noise_width)
    wn = np.random.randn(
        int(wn_length / max(1, noise_width))) * scaler
    wn = wn.repeat(noise_width)
    if pad > 0:
        wn = np.concatenate([wn, np.zeros(pad)])
    return wn
