import numpy as np
from scipy.signal import windows

class AddBreathingSound:

    mean_breath_duration = 4 # 12 - 20 breaths per minute
    std_breath_duration = 1

    def __init__(self, max_breathing_scale: float, freq: int):

        self.max_breathing_scale = max_breathing_scale
        self.freq = freq

    def _make_breathing(self, total_samples: int):

        breath_duration_s = np.random.normal(
            self.mean_breath_duration, self.std_breath_duration, 1)[0]
        breath_duration_s = np.clip(breath_duration_s, 1, 8)
        breath_samples = int(breath_duration_s * self.freq)

        # Generate white noise
        white_noise = np.random.normal(0, 1, breath_samples)

        # Create a sine wave for modulation
        t = np.linspace(0, breath_duration_s, breath_samples, endpoint=False)
        modulation = 0.5 * (1 + np.sin(2 * np.pi * t / breath_duration_s))

        # Apply a window to fade in and fade out the white noise (inhale and exhale)
        window = windows.tukey(breath_samples, alpha=0.5)
        inhale_exhale = white_noise * modulation * window

        # Repeat the inhale and exhale cycle
        num_cycles = int(total_samples // breath_samples)
        breathing_sound = np.tile(inhale_exhale, num_cycles)
        breathing_sound = breathing_sound / np.max(np.abs(breathing_sound))

        return breathing_sound

    def __call__(self, sample):

        data = sample["data"]
        breath = self._make_breathing(int(len(data) * 5))
        breath = breath * np.random.rand() * self.max_breathing_scale
        # align length with data by randomly cutting
        # find the start point of the cutting point
        if len(data) < len(breath):
            start = np.random.choice(len(breath) - len(data), 1)[0]
            data = data + breath[start:start+len(data)]
        sample.update(data=data)
        return sample

class RandomMask:

    def __init__(self, mask_ratio: float):

        self.mask_ratio = mask_ratio

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (data_length, )}
        Returns:
            sample (Dict): {"data": Array of shape (data_length, )}
        """
        data = sample["data"]
        mask_width = int(data.shape[0] * self.mask_ratio)
        mask_start = np.random.choice(data.shape[0] - mask_width, 1)[0]

        masked_data = data.copy()
        masked_data[mask_start:mask_start+mask_width] = 0

        sample.update(data=masked_data)
        return sample

class RandomMask:

    def __init__(self, mask_ratio: float):

        self.mask_ratio = mask_ratio

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (data_length, )}
        Returns:
            sample (Dict): {"data": Array of shape (data_length, )}
        """
        data = sample["data"]
        mask_width = int(data.shape[0] * self.mask_ratio)
        mask_start = np.random.choice(data.shape[0] - mask_width, 1)[0]

        masked_data = data.copy()
        masked_data[mask_start:mask_start+mask_width] = 0

        sample.update(data=masked_data)
        return sample

class RandomShift:

    def __init__(self, max_shift_ratio: float):

        self.max_shift_ratio = max_shift_ratio

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (data_length, )}
        Returns:
            sample (Dict): {"data": Array of shape (data_length, )}
        """
        data = sample["data"]
        shift_ratio = np.random.rand() * self.max_shift_ratio
        shift_size = int(data.shape[0] * shift_ratio)

        pad = np.zeros(shift_size)

        shifted_data = data.copy()
        if np.random.rand() < 0.5:
            shifted_data = np.concatenate([pad, shifted_data])[:len(data)]
        else:
            shifted_data = np.concatenate([shifted_data, pad])[-len(data):]
        assert len(data) == len(shifted_data)
        sample.update(data=shifted_data)
        return sample

class RandomFlip:

    def __init__(self, flip_rate: float):

        self.flip_rate = flip_rate

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (data_length, )}
        Returns:
            sample (Dict): {"data": Array of shape (data_length, )}
        """
        data = sample["data"]
        if np.random.rand() < self.flip_rate:
            data *= -1
        sample.update(data=data)
        return sample
    
class StretchSignal:

    def __init__(self, stretch_ratio: float):

        self.stretch_ratio = stretch_ratio
    
    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (data_length, )}
        Returns:
            sample (Dict): {"data": Array of shape (data_length, )}
        """
        data = sample["data"]
        stretch_ratio = 1 + (np.random.rand() * 2 - 1) * self.stretch_ratio
        new_len = int(len(data) * stretch_ratio)
        new_data = np.interp(
            np.linspace(0, 1, new_len, endpoint=False),
            np.linspace(0, 1, len(data), endpoint=False),
            data
        )

        if new_len == len(data):
            pass
        elif new_len > len(data):
            # randomly cut the signal to align with original length
            start = np.random.choice(new_len - len(data), 1)[0]
            new_data = new_data[start:start+len(data)]
        else:
            # randomly pad both sides of the signal to align with original length
            total_pad = np.zeros(len(data) - new_len)
            pad_l = np.random.choice(len(total_pad), 1)[0]
            new_data = np.concatenate([total_pad[:pad_l], new_data, total_pad[pad_l:]])
        sample.update(data=new_data)
        return sample
    
class RandomScaleSignal:

    def __init__(self, scale_ratio: float, scaler_freq: float=10.):

        self.scale_ratio = scale_ratio
        assert scaler_freq > 1
        self.scaler_freq = scaler_freq - 1

    def __call__(self, sample):
        """
        Args:
            sample (Dict): {"data": Array of shape (data_length, )}
        Returns:
            sample (Dict): {"data": Array of shape (data_length, )}
        """
        data = sample["data"]

        sine_values =\
            np.sin(2 * np.pi * \
                   np.linspace(0, 1, len(data), endpoint=False) \
            * np.random.rand() * self.scaler_freq + 1)

        # random scale factor
        scale_factor = np.random.rand() * self.scale_ratio
        min_scale_factor = 1 - scale_factor
        max_scale_factor = 1 + scale_factor

        scale_factors = sine_values * (max_scale_factor - min_scale_factor) + min_scale_factor
        scaled_data = data * scale_factors

        sample.update(data=scaled_data)
        return sample