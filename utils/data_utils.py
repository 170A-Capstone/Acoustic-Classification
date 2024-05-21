import os, time
import pandas as pd
import numpy as np
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
import numpy as np


from utils.sql_utils import DB
import utils.signal_processing_utils as sp

db = DB()

class Dataset():
    """Utilities for handling dataset
    """

    def __init__(self,directory_path,log=True) -> None:
        
        self.db = db
        self.directory_path = directory_path
        self.log = log

    def getFilePaths(self,filter_condition=None):
        paths = os.listdir(self.directory_path)

        if filter_condition:
            paths = [path for path in paths if filter_condition(path)]

        if self.log:
            print(f'[{self.log_label}]: Paths Acquired')

        return paths
    
    def uploadFeatures(self):
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
            print(f'[{self.log_label}]: Features Extracted')

        self.db.uploadDF(df,f'{self.log_label}_features')
        
        if self.log:
            b = time.time()
            print(f'[{self.log_label}]: Features Uploaded ({b-a:.2f}s)')

    def extractSignal(self,path,transform_func=None):
        """Extract one audio channel from .wav file

        Args:
            path (str): path to .wav file
            left (bool, optional): utilize the left audio channel if true, or else right channel. Defaults to True.
            compression_factor (_type_, optional): compression factor. Defaults to None, meaning no compression.

        Returns:
            array[int]: audio time-series data
        """

        samplerate, signal = wavfile.read(self.directory_path + path)

        if transform_func:
            signal = transform_func(signal)

        # cursor execute requires array to be contiguous
        signal = np.ascontiguousarray(signal)

        return signal

    def uploadSignals(self):

        if self.log:
            a = time.time()

        signals = [self.extractSignal(path) for path in self.paths]
        
        if self.log:
            b = time.time()
            print(f'[{self.log_label}]: Signals Extracted ({b-a:.2f}s)')

        self.db.uploadBLObs(signals,f'{self.log_label}_signals')
        
        if self.log:
            c = time.time()
            print(f'[{self.log_label}]: Signals Uploaded ({c-b:.2f}s)')
    
    def extractLabelEmbedding(self,class_label):
        embedding = [1 if class_ == class_label else 0 for class_ in self.classes]
        return embedding

    def downloadSignals(self):

        query_str = f'''
            SELECT signal 
            FROM "{self.log_label}_signals"
            '''

        if self.log:
            a = time.time()

        # query data
        self.db.cursor.execute(query_str)
        data = self.db.cursor.fetchall()

        if self.log:
            b = time.time()
            print(f'[{self.log_label}]: Data Queried ({b-a:.2f}s)')

        # decode BLObs to signals
        data = [np.frombuffer(blob, dtype=self.dtype) for blob, in data]

        if self.log:
            c = time.time()
            print(f'[{self.log_label}]: Data Decoded ({c-b:.2f}s)')

        return data
  
    def transformSignals(self,transform):

        transform_func = None
        columns = []
        table_name = f'{transform}_features'

        if transform == 'statistical':
            transform_func = sp.extractStatisticalFeatures
            columns = ['mode_var','k','s','mean','i','g','h','dev','var','variance','std','gstd_var','ent']

        elif transform == 'harmonic':
            transform_func = sp.extractHarmonicFeatures
            columns = ['dominant_freq', 'dominant_amplitude', 'first_harmonic_freq']


        signals = self.downloadSignals()

        if self.log:
            a = time.time()

        transformed_signals = [transform_func(signal) for signal in signals]

        # FOR DEBUGGING
        # for index,signal in enumerate(signals):
        #     try:
        #         transform_func(signal)
        #     except:
        #         print('index: ',index)

        if self.log:
            b = time.time()
            print(f'[{self.log_label}]: Data Transformed ({b-a:.2f}s)')

        df = pd.DataFrame(transformed_signals,columns=columns)

        self.db.uploadDF(df,f'{self.log_label}_{table_name}')

        if self.log:
            c = time.time()
            print(f'[{self.log_label}]: Features Uploaded ({c-b:.2f}s)')

    def getQueryStr(self,feature_set_type):

        # CANNOT USE BECAUSE FEATURES ARE CATEGORICAL

        if feature_set_type == 'ambient':
            return f'''
                SELECT speed,daytime,weather,"source direction",class 
                FROM "IDMT_features"
                limit 10
                '''
        if feature_set_type == 'statistical':
            return f'''
                SELECT stat_features.*,features.class 
                FROM "{self.log_label}_statistical_features" AS stat_features 
                LEFT JOIN "{self.log_label}_features" AS features 
                ON features.index = stat_features.index
                '''
        if feature_set_type == 'statistical-PCA':
            return f'''
                SELECT stat_features.*,features.class 
                FROM (
                    SELECT index,mode_var, s, g, var, gstd_var, ent 
                    FROM "{self.log_label}_statistical_features"
                ) AS stat_features 
                LEFT JOIN "{self.log_label}_features" AS features 
                ON features.index = stat_features.index
                '''
        if feature_set_type == 'bs':
            return f'''
                SELECT stat_features.*,features.class 
                FROM (
                    SELECT index,mode_var 
                    FROM "{self.log_label}_statistical_features"
                ) AS stat_features 
                LEFT JOIN "{self.log_label}_features" AS features 
                ON features.index = stat_features.index
                '''
        
        
    def constructDataLoader(self,feature_set_type):
        
        query_str = self.getQueryStr(feature_set_type)

        if self.log:
            a = time.time()

        self.db.cursor.execute(query_str)
        data = self.db.cursor.fetchall()

        # subtract index and class column to isolate training features
        feature_size = len(data[0])-2

        if self.log:
            b = time.time()
            print(f'[{self.log_label}]: Data Queried ({b-a:.2f}s)')

        # data = [(row[1:-1],self.extractLabelEmbedding(row[-1])) for row in data]

        # scale up values by order(s) of magnitude
        data = [(np.array(row[1:-1])*(10e5),self.extractLabelEmbedding(row[-1])) for row in data]

        # Split the array into train and test sets
        train_data, test_data = train_test_split(data, test_size=0.2)

        if self.log:
            c = time.time()
            print(f'[{self.log_label}]: Data Transformed ({c-b:.2f}s)')
        
        return feature_size,train_data,test_data
    
class IDMT(Dataset):
    def __init__(self,directory_path = "./IDMT_Traffic/audio/",log=True) -> None:
        super().__init__(directory_path,log)

        self.log_label = 'IDMT'
        self.columns = ['date','location','speed','position','daytime','weather','class','source direction','mic','channel']
        self.classes = ['B', 'C', 'M', 'T']
        self.dtype = np.float32

        self.paths = self.getFilePaths()

    def getFilePaths(self):

        # omit:
        #   1. background noise entries
        #   2. non-SE entries
        filter_condition = lambda path: '-BG' not in path and 'ME' not in path

        return super().getFilePaths(filter_condition=filter_condition)
    
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

    def extractSignal(self,path):

        # stereo -> left channel
        transform_func = lambda signal: signal[:,0]

        signal = super().extractSignal(path,transform_func)

        return signal

class IDMT_BG(Dataset):
    def __init__(self,directory_path = "./IDMT_Traffic/audio/",log=True) -> None:
        super().__init__(directory_path,log)

        self.log_label = 'IDMT-BG'
        self.columns = ['date','location','speed','position','mic','channel']
        
        # signal v noise
        self.classes = ['S', 'N']
        self.dtype = np.float32

        self.paths = self.getFilePaths()

    def getFilePaths(self):

        # filter:
        #   1. non-background noise entries
        #   2. non-SE entries
        filter_condition = lambda path: '-BG' in path and 'ME' not in path

        return super().getFilePaths(filter_condition=filter_condition)
    
    def extractFeatures(self,path):
        features = path[:-7].split('_')
        return features

    def extractSignal(self,path):

        # stereo -> left channel
        transform_func = lambda signal: signal[:,0]

        signal = super().extractSignal(path,transform_func)

        return signal

    def constructDataLoader(self):
        
        # query_str = self.getQueryStr(feature_set_type)
        # query_str = f'''
        #     select * from "IDMT_harmonic_features"
        #     limit 10
        #     '''

        # self.db.cursor.execute(query_str)
        # data = self.db.cursor.fetchall()
        

        if self.log:
            a = time.time()

        data1 = self.db.downloadDF('IDMT_harmonic_features')
        data2 = self.db.downloadDF('IDMT-BG_harmonic_features')
        
        if self.log:
            b = time.time()
            print(f'[{self.log_label}]: Data Queried ({b-a:.2f}s)')

        # data1.drop('index', axis=1)
        data1['label'] = 1

        # data2.drop('index', axis=1)
        data2['label'] = 0

        data = pd.concat([data1,data2])
        data = [(row[1:-1],[row[-1]]) for row in data.values]

        if self.log:
            c = time.time()
            print(f'[{self.log_label}]: Data Transformed ({c-b:.2f}s)')

        feature_size = 3

        return feature_size,data

class MVD(Dataset):
    def __init__(self,directory_path = "./MVDA/",log=True) -> None:
        super().__init__(directory_path,log)

        self.log_label = 'MVD'
        self.columns = ['record_num', 'mic', 'class']
        self.classes = ['N', 'C', 'M', 'T']
        self.dtype = np.int16

        self.paths = self.getFilePaths()

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