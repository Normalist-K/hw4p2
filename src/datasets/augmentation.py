import random

import numpy as np

from audiomentations.core.transforms_interface import BaseSpectrogramTransform


class AddGaussianNoise(BaseSpectrogramTransform):
    """Add gaussian noise to the samples"""
    """
    Code from audiomentations
    I modified AddGaussianNoise input from waveform to spectrogram
    """

    supports_multichannel = True

    def __init__(self, min_amplitude=0.001, max_amplitude=0.015, p=0.5):
        super().__init__(p)
        assert min_amplitude > 0.0
        assert max_amplitude > 0.0
        assert max_amplitude >= min_amplitude
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

    def randomize_parameters(self, samples):
        super().randomize_parameters(samples)
        if self.parameters["should_apply"]:
            self.parameters["amplitude"] = random.uniform(
                self.min_amplitude, self.max_amplitude
            )

    def apply(self, samples):
        noise = np.random.randn(*samples.shape).astype(np.float32)
        samples = samples + self.parameters["amplitude"] * noise
        return samples