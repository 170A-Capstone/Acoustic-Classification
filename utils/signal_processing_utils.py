
# import dependencies
# import librosa
# import librosa.display
import numpy as np
# from librosa.util import fix_length
import scipy as sp
from scipy.stats import kurtosis,skew,mode,gstd,describe,iqr,gmean,hmean,median_abs_deviation,variation,tstd,gstd,moment,entropy
from sklearn.preprocessing import normalize


def extractStatisticalFeatures(signal):

    #load the file (audio)

    # sig, sr = librosa.load(self.directory_path+file)
    
    # pad audio files to avoid any dimensional issue
    
    # required_audio_size=3
    # audio = fix_length(sig, size=required_audio_size*sr) 
    
    #Code for feature extraction

    # local feature extraction (use only one feature as per requirement)
    
    #S = gfcc(sig=sig, fs=sr, num_ceps=40,nfilts=128,nfft=2048,win_hop=0.0232,win_len=0.0464)
    #     S = mfcc(sig=sig, fs=sr, num_ceps=40,nfilts=128,nfft=2048,win_hop=0.0232,win_len=0.0464)
    #     S=librosa.feature.melspectrogram(sig, sr=sr, n_mels=128,n_fft=2048,hop_length=512,win_length=1024)

    # Global feature extraction
    
    ft=sp.fft.fft(signal) # code for computing the spectrum using Fast Fourier Transform
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

    gstd_var=gstd(spec)
    ent= entropy(spec)
    
    features=[mode_var,k,s,mean,i,g,h,dev,var,variance,std,gstd_var,ent]

    features=normalize([features])
    features=np.array(features)
    features=np.reshape(features,(13,))
    #feat=features # save the matrix and vector in a list
    
    return features




