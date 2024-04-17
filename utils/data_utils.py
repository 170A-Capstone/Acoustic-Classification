import os
import pandas as pd

import utils.preprocessing_utils as preproc

class IDMT():
    """Utilities for handling IDMT dataset
    """

    def __init__(self,log=False,compression_factor=100,directory_path = "./IDMT_Traffic/audio/") -> None:
        
        self.columns = ['date','location','speed','position','daytime','weather','class','source direction','mic','channel']
        self.classes = ['B', 'C', 'M', 'T']

        self.log = log
        self.compression_factor = compression_factor
        self.directory_path = directory_path

        if self.log:
            print('[IDMT]: IDMT Dataset Handler initialized')
        

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
    
    def getFeatureDF(self,paths):
        """Packages all features of dataset into dataframe

        Args:
            paths (array[str]): list of dataset paths

        Returns:
            pandas.DataFrame: dataframe containing dataset features
        """
        
        features = [self.extractFeatures(path) for path in paths]
        df = pd.DataFrame(features,columns=self.columns)

        if self.log:
            print('[IDMT]: Features Extracted')

        return df

    def extractAudio(self,relative_path):
        root_path = self.directory_path + relative_path
        return preproc.extractAudio(root_path,left=True,compression_factor=self.compression_factor)
    
    # rename getAudioDF to comply with getFeatureDF naming convention
    def extractAudioDF(self,paths):
        audio_data = [self.extractAudio(path) for path in paths]
        df = pd.DataFrame(audio_data)
        
        if self.log:
            print('[IDMT]: Audio Extracted')

        return df
    
    def extractLabelEmbedding(self,path):
        features = self.extractFeatures(path)
        label = features[self.columns.index('class')]
        embedding = [1 if class_ == label else 0 for class_ in self.classes]
        return embedding

    def constructDataLoader(self,paths):
        # construct training dataset
        loader = []
        for path in paths:
            audio = self.extractAudio(path)
            fft,compressed_fft = preproc.process(audio)

            label_embedding = self.extractLabelEmbedding(path)

            loader.append((compressed_fft,label_embedding))
        
        if self.log:
            print('[IDMT]: Train Loader Constructed')

        return loader



class MVD():
    def __init__(self,log=False,compression_factor=100,directory_path = "./MVDA/") -> None:
        
        self.columns = ['record_num', 'mic', 'class']
        self.classes = ['B', 'C', 'M', 'T']

        self.log = log
        self.compression_factor = compression_factor
        self.directory_path = directory_path

        if self.log:
            print('[MVD]: MVD Dataset Handler initialized')
        

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
    
    def getFeatureDF(self,paths):
        """Packages all features of dataset into dataframe

        Args:
            paths (array[str]): list of dataset paths

        Returns:
            pandas.DataFrame: dataframe containing dataset features
        """
        
        features = [self.extractFeatures(path) for path in paths]
        df = pd.DataFrame(features,columns=self.columns)

        if self.log:
            print('[MVD]: Features Extracted')

        return df

    def extractAudio(self,relative_path):
        root_path = self.directory_path + relative_path
        return preproc.extractAudio(root_path,left=True,compression_factor=self.compression_factor)
    
    # rename getAudioDF to comply with getFeatureDF naming convention
    def extractAudioDF(self,paths):
        audio_data = [self.extractAudio(path) for path in paths]
        df = pd.DataFrame(audio_data)
        
        if self.log:
            print('[MVD]: Audio Extracted')

        return df
    
    def extractLabelEmbedding(self,path):
        features = self.extractFeatures(path)
        label = features[self.columns.index('class')]
        embedding = [1 if class_ == label else 0 for class_ in self.classes]
        return embedding

    def constructDataLoader(self,paths):
        # construct training dataset
        loader = []
        for path in paths[:10]:
            audio = self.extractAudio(path)
            fft,compressed_fft = preproc.process(audio)

            label_embedding = self.extractLabelEmbedding(path)

            loader.append((compressed_fft,label_embedding))
        
        if self.log:
            print('[MVD]: Data Loader Constructed')

        return loader
