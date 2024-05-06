import numpy as np
import scipy as spy
from scipy.stats import kurtosis, skew, mode, gstd, describe, iqr, gmean, hmean, median_abs_deviation, variation, tstd, \
    gstd, moment, entropy
from sklearn.preprocessing import normalize
import librosa
from librosa import feature


def extractStatisticalFeatures(signal):
    ft = np.fft.fft(signal)
    magnitude = np.absolute(ft)
    spec = magnitude
    k = kurtosis(spec)
    s = skew(spec)
    mean = np.mean(spec)
    z = np.array(mode(spec)[0])
    mode_var = float(z)
    i = iqr(spec)
    g = gmean(spec)
    h = hmean(spec)
    dev = median_abs_deviation(spec)
    var = variation(spec)
    variance = np.var(spec)
    std = tstd(spec)

    # transform fails for the following indices of MVD dataset: 2188,4119,6122,7899,12687
    gstd_var = 0
    try:
        gstd_var = gstd(spec)
    except:
        pass

    ent = entropy(spec)

    features = [mode_var, k, s, mean, i, g, h, dev, var, variance, std, gstd_var, ent]

    features = normalize([features])
    features = np.array(features)
    features = np.reshape(features, (13,))
    # feat=features # save the matrix and vector in a list

    return features

def librosaextractFeatures(signal):

    ft = np.fft.fft(signal)

    frequency_coefficients = np.mean(np.fft.fftfreq(len(ft))) #提取音频的频率
    fundamental_frequency, voiced_flag, voiced_probs = librosa.pyin(signal.astype('float'), fmin=librosa.note_to_hz('C2'),
                                                 fmax=librosa.note_to_hz('C7')) #fundamental_frequency：基础音频

    if np.isnan(fundamental_frequency).sum() == len(fundamental_frequency):
        fundamental_frequency = np.nan_to_num(fundamental_frequency)
    else:
        fundamental_frequency = np.where(np.isnan(fundamental_frequency) == True, np.nanmean(fundamental_frequency), fundamental_frequency)

    short_time_energy = np.mean(feature.rms(y=signal)) # 短时能量
    average_zero_cross_rate = np.mean(feature.zero_crossing_rate(y=signal.astype('float'))) #过零率
    pitch_frequency = np.mean(69+12 * np.where(np.log2(fundamental_frequency/440) == -float('inf'), 0, np.log2(abs(fundamental_frequency)/440)))
    # 音高，把音高为负无穷，即基础频率为0的地方的音高替换成0
    fundamental_frequency = np.mean(fundamental_frequency) #取基础音频的平均值

    features = [frequency_coefficients, fundamental_frequency, short_time_energy
                , average_zero_cross_rate, pitch_frequency]

    features = normalize([features])
    features = np.array(features)
    features = np.reshape(features, (5,))
    # feat=features # save the matrix and vector in a list

    return features
