import torch
import torch.nn as nn
import torch.nn.functional as F

class Shallow(nn.Module):
    """ResNet model

    outputs 4 numbers, corresponding to a vector of probabilities for each possible class (car, truck, etc)
    """

    def __init__(self,input_dim,output_dim,log = False):

        self.log = log

        self.name = 'Shallow'

        super(Shallow, self).__init__()
        
        self.fc = nn.Linear(input_dim, output_dim)

        if self.log:
            print('[Model-]: Model initialized')
    
    def predict(self,inputs):
        inputs = torch.Tensor(inputs)
        return self.forward(inputs)

    def forward(self, x):
        x = self.fc(x)
        x = F.sigmoid(x)
        return x
    
    @staticmethod
    def name():
        return 'shallow'

class Deep(nn.Module):
    def __init__(self,input_dim,output_dim,log = False):

        super(Deep, self).__init__()

        layer_dim = int(input_dim/2)

        self.fc = nn.Linear(input_dim, layer_dim)
        self.shallow = Shallow(input_dim=layer_dim,output_dim=output_dim)
        
    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        return self.shallow(x)
    
    @staticmethod
    def name():
        return 'deep'
    
def loadModelParams(model,file_name):
    path = f'./model_params/{file_name}.pt'
    model.load_state_dict(torch.load(path))

    return model