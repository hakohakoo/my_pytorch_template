import math
import random
import time

import numpy as np


def get_noise_by_snr(signal, snr):
    SNR = snr
    noise = np.random.randn(signal.shape[0], signal.shape[1])
    signal_power = np.linalg.norm(signal) ** 2 / signal.size
    noise_variance = signal_power / np.power(10, (SNR / 10))
    noise = (np.sqrt(noise_variance) / np.std(noise)) * noise
    signal_noise = noise + signal

    Ps = (np.linalg.norm(signal)) ** 2  # signal power
    Pn = (np.linalg.norm(signal - signal_noise)) ** 2  # noise power
    snr = 10 * np.log10(Ps / Pn)
    return signal_noise, noise