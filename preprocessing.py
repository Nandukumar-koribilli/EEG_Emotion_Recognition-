import numpy as np
from scipy.signal import butter, lfilter

# Bandpass filter
def bandpass_filter(data, lowcut=4, highcut=45, fs=128, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype='band')
    filtered_data = lfilter(b, a, data)

    return filtered_data


# Normalize EEG signal
def normalize_data(data):
    mean = np.mean(data)
    std = np.std(data)

    normalized = (data - mean) / std

    return normalized