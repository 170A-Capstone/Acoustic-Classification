import numpy as np
from scipy.io import wavfile

def compressVector(vector,compression_factor):
    """Compress audio time-series array

    Args:
        vector (array[double]): uncompressed time-series data
        compression_factor (int): compression factor

    Returns:
        array[double]: compressed time-series data of length len(vector) / compression_factor
    """

    # Calculate the number of bins
    num_bins = len(vector) // compression_factor

    # Reshape the array into a 2D array with shape (num_bins, compression_factor)
    reshaped_array = vector[:num_bins * compression_factor].reshape((num_bins, compression_factor))

    # Calculate the average of each bin
    return np.mean(reshaped_array, axis=1)

def extractAudio(path,left = True,compression_factor = None):
    """Extract one audio channel from .wav file

    Args:
        path (str): path to .wav file
        left (bool, optional): utilize the left audio channel if true, or else right channel. Defaults to True.
        compression_factor (_type_, optional): compression factor. Defaults to None, meaning no compression.

    Returns:
        array[int]: audio time-series data
    """
    samplerate, time_data = wavfile.read(path)


    if len(time_data.shape) == 2:
    # Check the audio file is stereo or mono
        data_channel = time_data[:,0 if left else 1]
    else:
        data_channel = time_data

    if compression_factor:
        return compressVector(data_channel,100)
    else:
        return data_channel
    
# class Audio():
#     def __init__(self) -> None:
#         pass

    
def process(audio):

    # https://www.youtube.com/watch?v=aQKX3mrDFoY

    fft = np.fft.fft(audio)
    fft = np.abs(fft)[0:int(len(audio)/2)]
    compressed_fft = compressVector(fft,compression_factor=100)
    return fft,compressed_fft