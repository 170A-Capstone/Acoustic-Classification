import os
import pandas as pd

import utils.preprocessing_utils as preproc

class IDMT():
    """Utilities for handling IDMT dataset
    """

    def __init__(self,log=False,compression_factor=100,directory_path = "./IDMT_Traffic/audio/") -> None:
        self.columns = ['date','location','speed','position','daytime','weather','class','source direction','mic','channel']
        
        self.log = log
        self.compression_factor = compression_factor
        self.directory_path = directory_path

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
    
    def extractAudioDF(self,paths):
        audio_data = [self.extractAudio(path) for path in paths]
        df = pd.DataFrame(audio_data)
        
        if self.log:
            print('[IDMT]: Audio Extracted')

        return df

class MVD():
    def __init__(self) -> None:
        pass