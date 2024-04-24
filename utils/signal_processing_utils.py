import numpy as np
import scipy as sp
from scipy.stats import kurtosis,skew,mode,gstd,describe,iqr,gmean,hmean,median_abs_deviation,variation,tstd,gstd,moment,entropy
from sklearn.preprocessing import normalize


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

    # transform fails for the following indices of MVD dataset: 2188,4119,6122,7899,12687
    gstd_var = 0
    try:
        gstd_var=gstd(spec)
    except:
        pass

    ent= entropy(spec)
    
    features=[mode_var,k,s,mean,i,g,h,dev,var,variance,std,gstd_var,ent]

    features=normalize([features])
    features=np.array(features)
    features=np.reshape(features,(13,))
    #feat=features # save the matrix and vector in a list
    
    return features




