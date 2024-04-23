import os
import pandas as pd
import time

import utils.preprocessing_utils as preproc
import utils.signal_processing_utils as sp

class IDMT():
    """Utilities for handling IDMT dataset
    """

    def __init__(self,db,directory_path = "./IDMT_Traffic/audio/",log=True) -> None:
        
        self.db = db
        self.directory_path = directory_path
        self.log = log

        self.columns = ['date','location','speed','position','daytime','weather','class','source direction','mic','channel']
        self.classes = ['B', 'C', 'M', 'T']

        if self.log:
            print('[IDMT]: IDMT Dataset Handler initialized')

        self.paths = self.getFilePaths()
        
    def getFilePaths(self):
        paths = os.listdir(self.directory_path)

        # remove background noise entries
        paths = [path for path in paths if '-BG' not in path]

        # remove ME mic entries
        paths = [path for path in paths if 'ME' not in path]
        
        if self.log:
            print('[IDMT]: Paths Acquired')

        return paths

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
    
    def uploadFeatureDF(self):
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

        self.db.uploadDF(df,'idmt_features')
        
        if self.log:
            b = time.time()
            print(f'[IDMT]: Features Uploaded ({b-a:.2f}s)')

    def uploadSignalsDF(self,compression_factor=None):

        if self.log:
            a = time.time()

        # 78 seconds
        signals = [preproc.extractAudio(self.directory_path + path,left=True,compression_factor=compression_factor) for path in self.paths]
        
        if self.log:
            b = time.time()
            print(f'[IDMT]: Signals Extracted ({b-a:.2f}s)')

        # 117 seconds
        self.db.uploadBLObs(signals,'idmt_signals')
        
        if self.log:
            c = time.time()
            print(f'[IDMT]: Signals Uploaded ({c-b:.2f}s)')
    
    def extractLabelEmbedding(self,class_label):
        embedding = [1 if class_ == class_label else 0 for class_ in self.classes]
        return embedding

    def downloadSignals(self):

        query_str = f'''
            SELECT audio 
            FROM idmt_signals
            '''

        if self.log:
            a = time.time()

        # query data (98s)
        self.db.cursor.execute(query_str)
        data = self.db.cursor.fetchall()

        if self.log:
            b = time.time()
            print(f'[IDMT]: Data Queried ({b-a:.2f}s)')

        # decode signals (.02s)
        data = [preproc.decodeBLOb(blob[0]) for blob in data]

        if self.log:
            c = time.time()
            print(f'[IDMT]: Data Decoded ({c-b:.2f}s)')

        return data
  
    def transformSignals(self,transform):

        transform_func = None
        columns = []
        table_name = ''

        if transform == 'statistical':
            transform_func = sp.extractStatisticalFeatures
            columns = ['mode_var','k','s','mean','i','g','h','dev','var','variance','std','gstd_var','ent']
            table_name = 'statistical_features'


        signals = self.downloadSignals()

        if self.log:
            a = time.time()

        transformed_signals = [transform_func(signal) for signal in signals]

        if self.log:
            b = time.time()
            print(f'[IDMT]: Data Transformed ({b-a:.2f}s)')

        df = pd.DataFrame(transformed_signals,columns=columns)
        self.db.uploadDF(df,f'idmt_{table_name}')

        if self.log:
            c = time.time()
            print(f'[IDMT]: Features Uploaded ({c-b:.2f}s)')

    def constructDataLoader(self):
        query_str = f'''
            SELECT stat_features.*,features.class 
            FROM idmt_statistical_features AS stat_features 
            LEFT JOIN idmt_features AS features 
            ON features.index = stat_features.index
            '''

        if self.log:
            a = time.time()

        self.db.cursor.execute(query_str)
        data = self.db.cursor.fetchall()

        if self.log:
            b = time.time()
            print(f'[IDMT]: Data Queried ({b-a:.2f}s)')

        data = [(row[1:-1],self.extractLabelEmbedding(row[-1])) for row in data]

        if self.log:
            c = time.time()
            print(f'[IDMT]: Data Transformed ({c-b:.2f}s)')
        
        return data
    

class MVD():
    def __init__(self) -> None:
        pass