import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch

# Extract time-domain features
def extract_time_features(signal):

    mean = np.mean(signal)
    variance = np.var(signal)
    std = np.std(signal)
    rms = np.sqrt(np.mean(signal**2))
    sk = skew(signal)
    kurt = kurtosis(signal)

    return [mean, variance, std, rms, sk, kurt]


# Extract frequency-domain features
def extract_frequency_features(signal, fs=128):

    freqs, psd = welch(signal, fs)

    alpha = np.mean(psd[(freqs >= 8) & (freqs <= 13)])
    beta = np.mean(psd[(freqs >= 13) & (freqs <= 30)])
    theta = np.mean(psd[(freqs >= 4) & (freqs <= 8)])
    gamma = np.mean(psd[(freqs >= 30) & (freqs <= 45)])

    return [alpha, beta, theta, gamma]


# Differential Entropy
def differential_entropy(signal):

    variance = np.var(signal)

    de = 0.5 * np.log(2 * np.pi * np.e * variance)

    return de


# Combine all features
def extract_features(signal):

    time_features = extract_time_features(signal)

    freq_features = extract_frequency_features(signal)

    de_feature = [differential_entropy(signal)]

    features = time_features + freq_features + de_feature

    return features