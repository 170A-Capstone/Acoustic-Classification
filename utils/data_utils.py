import os, time

import librosa
from librosa import feature
import pandas as pd
import numpy as np
from scipy.io import wavfile

import signal_processing_utils as sp





class Dataset():
    """Utilities for handling dataset
    """

    def __init__(self, db, directory_path, log=True) -> None:

        self.db = db
        self.directory_path = directory_path
        self.log = log

    def getFilePaths(self, filter_condition=None):
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
        df = pd.DataFrame(features, columns=self.columns)

        if self.log:
            a = time.time()
            print(f'[{self.log_label}]: Features Extracted')

        self.db.uploadDF(df, f'{self.log_label}_features')

        if self.log:
            b = time.time()
            print(f'[{self.log_label}]: Features Uploaded ({b - a:.2f}s)')

    def extractSignal(self, path, transform_func=None):
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
            print(f'[{self.log_label}]: Signals Extracted ({b - a:.2f}s)')

        self.db.uploadBLObs(signals, f'{self.log_label}_signals')

        if self.log:
            c = time.time()
            print(f'[{self.log_label}]: Signals Uploaded ({c - b:.2f}s)')

    def extractLabelEmbedding(self, class_label):
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
            print(f'[{self.log_label}]: Data Queried ({b - a:.2f}s)')

        # decode BLObs to signals
        data = [np.frombuffer(blob, dtype=self.dtype) for blob, in data]

        if self.log:
            c = time.time()
            print(f'[{self.log_label}]: Data Decoded ({c - b:.2f}s)')

        return data

    def transformSignals(self, transform):

        transform_func = None
        columns = []
        table_name = ''

        if transform == 'statistical':
            transform_func = sp.extractStatisticalFeatures
            columns = ['mode_var', 'k', 's', 'mean', 'i', 'g', 'h', 'dev', 'var', 'variance', 'std', 'gstd_var', 'ent']
            table_name = 'statistical_features'
        else:
            transform_func = sp.librosaextractFeatures
            columns = ['frequency_coefficients', 'fundamental_frequency', 'short_time_energy'
                , 'average_zero_cross_rate', 'pitch_frequency']
            table_name = 'librosa_features'

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
            print(f'[{self.log_label}]: Data Transformed ({b - a:.2f}s)')

        df = pd.DataFrame(transformed_signals, columns=columns)
        self.db.uploadDF(df, f'{self.log_label}_{table_name}')

        if self.log:
            c = time.time()
            print(f'[{self.log_label}]: Features Uploaded ({c - b:.2f}s)')

    def constructDataLoader(self, transform):
        if transform == 'statistical':
            query_str = f'''
                SELECT stat_features.*,features.class 
                FROM "{self.log_label}_statistical_features" AS stat_features 
                LEFT JOIN "{self.log_label}_features" AS features 
                ON features.index = stat_features.index
                '''
        else:
            query_str = f'''
                SELECT librosa_features.*,features.class 
                FROM "{self.log_label}_librosa_features" AS librosa_features 
                LEFT JOIN "{self.log_label}_features" AS features 
                ON features.index = librosa_features.index
                '''

        if self.log:
            a = time.time()

        self.db.cursor.execute(query_str)
        data = self.db.cursor.fetchall()

        if self.log:
            b = time.time()
            print(f'[{self.log_label}]: Data Queried ({b - a:.2f}s)')

        data = [(row[1:-1], self.extractLabelEmbedding(row[-1])) for row in data]

        if self.log:
            c = time.time()
            print(f'[{self.log_label}]: Data Transformed ({c - b:.2f}s)')

        return data

class IDMT(Dataset):
    def __init__(self, db, directory_path="./IDMT_Traffic/audio/", log=True) -> None:
        super().__init__(db, directory_path, log)

        self.log_label = 'IDMT'
        self.columns = ['date', 'location', 'speed', 'position', 'daytime', 'weather', 'class', 'source direction',
                        'mic', 'channel']
        self.classes = ['B', 'C', 'M', 'T']
        self.dtype = np.float32

        self.paths = self.getFilePaths()

    def getFilePaths(self):
        # omit:
        #   1. background noise entries
        #   2. non-SE entries
        filter_condition = lambda path: '-BG' not in path and 'ME' not in path

        return super().getFilePaths(filter_condition=filter_condition)

    def extractFeatures(self, path):
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
        features = features[:6] + [features[6][0], features[6][1]] + features[7:]
        return features

    def extractSignal(self, path):
        # stereo -> left channel
        transform_func = lambda signal: signal[:, 0]

        signal = super().extractSignal(path, transform_func)

        return signal


class MVD(Dataset):
    def __init__(self, db, directory_path="./MVDA/", log=True) -> None:
        super().__init__(db, directory_path, log)

        self.log_label = 'MVD'
        self.columns = ['record_num', 'mic', 'class']
        self.classes = ['N', 'C', 'M', 'T']
        self.dtype = np.int16

        self.paths = self.getFilePaths()

    def extractFeatures(self, path):
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

