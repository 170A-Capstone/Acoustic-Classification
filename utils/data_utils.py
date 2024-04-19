import os
import pandas as pd
import time

import utils.preprocessing_utils as preproc

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
        non_noise = [path for path in paths if '-BG' not in path]
        
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
            print('[IDMT]: Features Extracted')

        self.db.uploadDF(df,table_name)
        
        if self.log:
            b = time.time()
            print(f'[IDMT]: Features Uploaded ({b-a}s)')

    def extractAudio(self,relative_path,compression_factor=None):
        root_path = self.directory_path + relative_path
        return preproc.extractAudio(root_path,left=True,compression_factor=compression_factor)
    
    # rename getAudioDF to comply with getFeatureDF naming convention
    def uploadAudioDF(self,table_name,compression_factor=None):

        if self.log:
            a = time.time()

        # 78 seconds
        audio_waveforms = [self.extractAudio(path,compression_factor) for path in self.paths]
        
        if self.log:
            b = time.time()
            print(f'[IDMT]: Audio Extracted ({b-a}s)')

        # 117 seconds
        self.db.uploadBLObs(audio_waveforms,table_name)
        
        if self.log:
            c = time.time()
            print(f'[IDMT]: Audio Uploaded ({c-b}s)')
    
    def extractLabelEmbedding(self,class_label):
        embedding = [1 if class_ == class_label else 0 for class_ in self.classes]
        return embedding

    def constructDataLoader(self):
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
    def __init__(self) -> None:
        pass