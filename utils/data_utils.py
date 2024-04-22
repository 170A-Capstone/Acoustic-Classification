import os
import pandas as pd
import psycopg2
import utils.preprocessing_utils as preproc
import time

# import dependencies
import tensorflow as tf
from glob import glob
import librosa
import librosa.display
import numpy as np
from librosa.util import fix_length
import scipy as sp
import spafe
from scipy.stats import kurtosis,skew,mode,gstd,describe,iqr,gmean,hmean,median_abs_deviation,variation,tstd,gstd,moment,entropy
from spafe.features import pncc,gfcc,bfcc,mfcc,lfcc,lpc,ngcc,rplp,psrcc,msrcc
from sklearn.preprocessing import normalize


class IDMT():
    """Utilities for handling IDMT dataset
    """

    def __init__(self,db,directory_path = "/Users/apple/Desktop/IDMT/audio/", log=True) -> None:
        
        self.db = db

        #what is log and compression_factor here

        self.columns = ['date','location','speed','position','daytime','weather','class','source direction','mic','channel']
        self.classes = ['B', 'C', 'M', 'T']
        self.audio_feature_columns = ['mode_var','kurtosis','skew','mean','iqr','gmean','hmean','median_abs_dev','variation','variance','std','gstd_var','ent']

        self.log = log
        self.directory_path = directory_path

        if self.log:
            print('[IDMT]: IDMT Dataset Handler initialized')
        
        self.paths = self.getFilePaths()
        

    def getFilePaths(self):
        paths = os.listdir(self.directory_path)
        non_noise = [path for path in paths if '-BG' not in path and path != "2019-10-22-15-30_Fraunhofer-IDMT_30Kmh_650690_A_D_CR_ME_CH12.wav"]
        
        if self.log:
            print('[IDMT]: Paths Acquired')

        return non_noise

    def extractFeatures(self,path):
        """Extracts known features embedded within file name of dataset instance

        Args:
            path (str): stringified features
                example: "2019-10-22-08-40_Fraunhofer-IDMT_30Kmh_1116695_M_D_CR_ME_CH12.wav"

        Returns:
            array[str]: itemized features
                example: ['2019-10-22-08-40', 'Fraunhofer-IDMT', '30', '1116695', 'M', 'D', 'C', 'R', 'ME', '12']
        """

        features = path[:-4].split('_')
        features[2] = features[2][:-3]
        features[-1] = features[-1][2:]
        features = features[:6] + [features[6][0],features[6][1]] + features[7:]
        return features
    
    def uploadFeatureDF(self, table_name):
        """Packages all features of dataset into dataframe

        Args:
            paths (array[str]): list of dataset paths

        Returns:
            pandas.DataFrame: dataframe containing dataset features
        """
        
        features = [self.extractFeatures(path) for path in self.paths]
        df = pd.DataFrame(features,columns=self.columns)

        if self.log:
            a = time.time()
            print('[IDMT]: Features Extracted')
        
        self.db.uploadDF(df, table_name)

        if self.log:
            b = time.time()
            print(f'[IDMT]: Features Uploaded ({b-a}s)')
        

    def extractAudio(self,relative_path,compression_factor=None):
        root_path = self.directory_path + relative_path
        return preproc.extractAudio(root_path,left=True,compression_factor=compression_factor)
    
    # rename getAudioDF to comply with getFeatureDF naming convention
    def uploadAudioDF(self,table_name, compression_factor=None):
        if self.log:
            a = time.time()

        audio_waveforms = [self.extractAudio(path, compression_factor) for path in self.paths]
        
        if self.log:
            b = time.time()
            print(f'[IDMT]: Audio Extracted ({b-a}s)')

        self.db.uploadBLObs(audio_waveforms, table_name)

        if self.log:
            c = time.time()
            print(f'[IDMT]: Audio Uploaded ({c-b}s)')
    
    def extractLabelEmbedding(self,class_label):
        embedding = [1 if class_ == class_label else 0 for class_ in self.classes]
        return embedding

    ########################################################################################################
    ########################################################################################################

    def audio_features_extractor(self, file):
    
        #load the file (audio)

        sig, sr = librosa.load(self.directory_path+file)
        
        # pad audio files to avoid any dimensional issue
        
        required_audio_size=3
        audio = fix_length(sig, size=required_audio_size*sr) 
        
        #Code for feature extraction

        # local feature extraction (use only one feature as per requirement)
        
        #S = gfcc(sig=sig, fs=sr, num_ceps=40,nfilts=128,nfft=2048,win_hop=0.0232,win_len=0.0464)
        #     S = mfcc(sig=sig, fs=sr, num_ceps=40,nfilts=128,nfft=2048,win_hop=0.0232,win_len=0.0464)
        #     S=librosa.feature.melspectrogram(sig, sr=sr, n_mels=128,n_fft=2048,hop_length=512,win_length=1024)

        # Global feature extraction
        
        ft=sp.fft.fft(sig) # code for computing the spectrum using Fast Fourier Transform
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
    
    def audio_features_DF(self, table_name):
        audio_features = [self.audio_features_extractor(path) for path in self.paths]
        df = pd.DataFrame(audio_features,columns=self.audio_feature_columns)

        if self.log:
            a = time.time()
            print('[IDMT]: Audio Features Extracted')

        self.db.uploadDF(df, table_name)

        if self.log:
            b = time.time()
            print(f'[IDMT]: Audio Features Uploaded ({b-a}s)')


    def constructDataLoader(self,paths):
        query_str = f'''
            SELECT audio.audio,features.class 
            FROM idmt_features AS features 
            LEFT JOIN idmt_audio AS audio 
            ON features.index = audio.index
            '''

            # order by features.index desc
            # limit 1000
        
        if self.log:
            a = time.time()

        # query data (98s)
        self.db.cursor.execute(query_str)
        data = self.db.cursor.fetchall()

        if self.log:
            b = time.time()
            print(f'[IDMT]: Data Queried ({b-a}s)')


        # decode audio waveforms (.02s)
        data = [(preproc.decodeBLOb(blob),class_label) for blob,class_label in data]

        if self.log:
            c = time.time()
            print(f'[IDMT]: Data Decoded ({c-b}s)')

        # feature engineering (91s)

        data = [(preproc.process(waveform),self.extractLabelEmbedding(class_label)) for waveform,class_label in data]

        if self.log:
            d = time.time()
            print(f'[IDMT]: Data Transformed ({d-c}s)')
        
        return data

class MVD():
    def __init__(self,db,log=True,directory_path = "/Users/apple/Desktop/MVD/") -> None:
        
        self.db = db

        self.columns = ['record_num', 'mic', 'class']
        self.classes = ['N', 'C', 'M', 'T']
        self.audio_feature_columns = ['mode_var','kurtosis','skew','mean','iqr','gmean','hmean','median_abs_dev','variation','variance','std','gstd_var','ent']

        self.log = log
        self.directory_path = directory_path

        if self.log:
            print('[MVD]: MVD Dataset Handler initialized')
        
        self.paths = self.getFilePaths()
        

    def getFilePaths(self): #required explaination
        paths = os.listdir(self.directory_path)
        non_noise = [path for path in paths if '-BG' not in path]
        
        if self.log:
            print('[MVD]: Paths Acquired')
        #print(non_noise)
        return non_noise

    def extractFeatures(self,path):
        """Extracts known features embedded within file name of dataset instance

        Args:
            path (str): stringified features
                example: "Recording_1_H_M.wav"

        Returns:
            array[str]: itemized features
                example: ['1', 'H', 'M']
        """

        features = path[:-4].split('_')
        features = features[1:]

        return features
    
    def uploadFeatureDF(self,table_name):
        """Packages all features of dataset into dataframe

        Args:
            paths (array[str]): list of dataset paths

        Returns:
            pandas.DataFrame: dataframe containing dataset features
        """
        
        features = [self.extractFeatures(path) for path in self.paths]
        df = pd.DataFrame(features,columns=self.columns)

        if self.log:
            a = time.time()
            print('[MVD]: Features Extracted')

        self.db.uploadDF(df, table_name)

        if self.log:
            b = time.time()
            print(f'[MVD]: Features Uploaded ({b-a}s)')


    def extractAudio(self,relative_path,compression_factor=None):
        root_path = self.directory_path + relative_path
        return preproc.extractAudio(root_path,left=True,compression_factor=compression_factor)
    
    # rename getAudioDF to comply with getFeatureDF naming convention
    def uploadAudioDF(self,table_name, compression_factor=None):
        
        if self.log:
            a = time.time()
        audio_waveforms = [self.extractAudio(path, compression_factor) for path in self.paths]
        df = pd.DataFrame(audio_waveforms)
        
        if self.log:
            b = time.time()
            print(f'[MVD]: Audio Extracted ({b-a}s)')

        self.db.uploadBLObs(audio_waveforms,table_name)
        
        if self.log:
            c = time.time()
            print(f'[MVD]: Audio Uploaded ({c-b}s)')
    
    def extractLabelEmbedding(self,class_label):
        embedding = [1 if class_ == class_label else 0 for class_ in self.classes]
        return embedding
    

    ########################################################################################################
    ########################################################################################################

    def audio_features_extractor(self, file):
    
        #load the file (audio)

        sig, sr = librosa.load(self.directory_path+file)
        
        # pad audio files to avoid any dimensional issue
        
        required_audio_size=3
        audio = fix_length(sig, size=required_audio_size*sr) 
        
        #Code for feature extraction

        # local feature extraction (use only one feature as per requirement)
        
        #S = gfcc(sig=sig, fs=sr, num_ceps=40,nfilts=128,nfft=2048,win_hop=0.0232,win_len=0.0464)
        #     S = mfcc(sig=sig, fs=sr, num_ceps=40,nfilts=128,nfft=2048,win_hop=0.0232,win_len=0.0464)
        #     S=librosa.feature.melspectrogram(sig, sr=sr, n_mels=128,n_fft=2048,hop_length=512,win_length=1024)

        # Global feature extraction
        
        ft=sp.fft.fft(sig) # code for computing the spectrum using Fast Fourier Transform
        magnitude=np.absolute(ft)
        spec=magnitude[0:11025]
        
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
        feat=features # save the matrix and vector in a list
        
        return feat
    
    def audio_features_DF(self, table_name):
        audio_features = [self.audio_features_extractor(path) for path in self.paths]
        df = pd.DataFrame(audio_features,columns=self.audio_feature_columns)

        if self.log:
            a = time.time()
            print('[MVD]: Audio Features Extracted')

        self.db.uploadDF(df, table_name)

        if self.log:
            b = time.time()
            print(f'[MVD]: Audio Features Uploaded ({b-a}s)')


    def constructDataLoader(self):
        query_str = f'''
            SELECT audio.audio,features.class 
            FROM mvd_features AS features 
            LEFT JOIN mvd_audio AS audio 
            ON features.index = audio.index
            '''
        # construct training dataset
        if self.log:
            a = time.time()

        # query data (98s)
        self.db.cursor.execute(query_str)
        data = self.db.cursor.fetchall()

        if self.log:
            b = time.time()
            print(f'[MVD]: Data Queried ({b-a}s)')


        # decode audio waveforms (.02s)
        data = [(preproc.decodeBLOb(blob),class_label) for blob,class_label in data]

        if self.log:
            c = time.time()
            print(f'[MVD]: Data Decoded ({c-b}s)')

        # feature engineering (91s)

        data = [(preproc.process(waveform),self.extractLabelEmbedding(class_label)) for waveform,class_label in data]

        if self.log:
            d = time.time()
            print(f'[MVD]: Data Transformed ({d-c}s)')
        
        return data
