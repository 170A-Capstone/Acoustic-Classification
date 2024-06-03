import numpy as np
import scipy as sp
from scipy.stats import kurtosis,skew,mode,gstd,describe,iqr,gmean,hmean,median_abs_deviation,variation,tstd,gstd,moment,entropy
from sklearn.preprocessing import normalize
import librosa
from librosa import feature


def extractStatisticalFeatures(signal):

    ft=sp.fft.fft(signal) 
    magnitude=np.absolute(ft)
    spec=magnitude
    
    k=kurtosis(spec)
    s=skew(spec)
    mean=np.mean(spec)
    z=np.array(mode(spec)[0])
    mode_var=float(z)
    i=iqr(spec)
    g=gmean(spec)
    h=hmean(spec)
    dev=median_abs_deviation(spec)
    var=variation(spec)
    variance=np.var(spec)
    std=tstd(spec)
    fc = np.mean(np.fft.fftfreq(len(ft)))
    azcr = np.mean(feature.zero_crossing_rate(y=signal.astype('float')))

    # transform fails for the following indices of MVD dataset: 2188,4119,6122,7899,12687
    gstd_var = 0
    try:
        gstd_var=gstd(spec)
    except:
        pass

    ent= entropy(spec)
    
    features=[mode_var,k,s,mean,i,g,h,dev,var,variance,std,gstd_var,ent,fc,azcr]

    features=normalize([features])
    features=np.array(features)
    features=np.reshape(features,(15,))
    #feat=features # save the matrix and vector in a list
    
    return features

SR = 9866

def extractHarmonicFeatures(signal):
    # Perform FFT on the signal
    fft_result = np.fft.fft(signal)
    
    # Get the frequencies corresponding to the FFT result
    frequencies = np.fft.fftfreq(len(signal), 1/SR)
    
    # Find the index of the maximum amplitude in the FFT result
    max_amp_index = np.argmax(np.abs(fft_result))

    # return max_amp_index
    
    # Extract the dominant frequency, first harmonic, and their amplitudes
    dominant_freq = frequencies[max_amp_index]
    dominant_amplitude = np.abs(fft_result[max_amp_index])
    first_harmonic_freq = dominant_freq * 2
    # first_harmonic_amplitude = np.abs(fft_result[max_amp_index * 2])

    features = [dominant_freq, dominant_amplitude, first_harmonic_freq]

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
    #pitch_frequency = np.mean(69+12*np.ma.log2(abs(np.array(fundamental_frequency))/440).filled(0))
    fundamental_frequency = np.mean(fundamental_frequency) #取基础音频的平均值

    features = [frequency_coefficients, short_time_energy, average_zero_cross_rate]#fundamental_frequency, average_zero_cross_rate]#, pitch_frequency]

    #features = normalize([features])
    features = np.array(features)
    features = np.reshape(features, (3,))
    # feat=features # save the matrix and vector in a list

    return features