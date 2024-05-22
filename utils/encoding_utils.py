from utils.data_utils import Dataset,IDMT
from models.nn import AutoEncoder,loadModelParams

import torch
import numpy as np

class IDMT_Encode(IDMT):
    def __init__(self,params_path,coding_layers,latent_dim) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        self.ae = AutoEncoder(input_dim=96001,cl=coding_layers,ld=latent_dim)

        # which one?
        # loadModelParams(model,file_name)
        # self.ae = loadModelParams(self.ae,params_path)

    def encode(self,signal):

        if len(signal) != 96001:
            signal = np.concatenate((signal,[0]))
        
        signal = torch.Tensor(np.array(signal))

        features = self.ae.code(signal,encode=True)
        
        return features.detach().numpy().tolist()

    def getTransformUtils(self,transform):

        transform_func = None
        columns = []

        if transform == 'encode':
            transform_func = self.encode
            columns = [f'param{i}' for i in range(self.latent_dim)]
        else:
            transform_func,columns = super().getTransformUtils(transform)

        return transform_func,columns
