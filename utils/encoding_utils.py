from utils.data_utils import Dataset,IDMT
from models.nn import ConvolutionalAutoEncoder,loadModelParams

import torch
import numpy as np

class IDMT_Encode(IDMT):
    def __init__(self,params_path,coding_layers,latent_dim) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.feature_size = 6001

        self.cae = ConvolutionalAutoEncoder(input_dim=self.feature_size,cl=coding_layers,ld=latent_dim)

        # which one?
        # loadModelParams(model,file_name)
        self.cae = loadModelParams(self.cae,params_path)

    def encode(self,signal):

        if len(signal) != self.feature_size:
            signal = np.concatenate((signal,[0]))

        signal = torch.Tensor(np.array(signal))

        signal = torch.reshape(signal,shape=(1,self.feature_size,1))

        features = self.cae.code(signal,encode=True)

        features = torch.reshape(features,shape=(1,self.latent_dim))
        
        return features.detach().numpy().tolist()[0]

    def getTransformUtils(self,transform):

        transform_func = None
        columns = []

        if transform == 'encode':
            transform_func = self.encode
            columns = [f'v{i}' for i in range(self.latent_dim)]
        else:
            transform_func,columns = super().getTransformUtils(transform)

        return transform_func,columns
