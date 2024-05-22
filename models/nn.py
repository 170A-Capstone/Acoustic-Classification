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
        # x = F.relu(x)
        return x
    
    @staticmethod
    def name():
        return 'shallow'

class Deep(nn.Module):
    def __init__(self,input_dim,output_dim,log = False):

        super(Deep, self).__init__()

        self.name = 'Deep'

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
  
class Deep2(nn.Module):
    def __init__(self,input_dim,output_dim,log = False):

        super(Deep2, self).__init__()

        self.name = 'Deep 2 HL'

        self.fc = nn.Linear(input_dim, input_dim)
        self.deep = Deep(input_dim=input_dim,output_dim=output_dim)
        
    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        return self.deep(x)
    
    @staticmethod
    def name():
        return 'deep w/ 2 HL'
      
class AutoEncoder(nn.Module):
    def __init__(self,input_dim,cl=1,ld=6,exponent=6,log = False):

        super(AutoEncoder, self).__init__()

        self.name = 'AutoEncoder'

        self.input_dim=input_dim
        self.ld=ld
        self.cl=cl
        self.exponent=exponent

        self.generateCodingSeries()

    def generateCodingSeries(self):
        layer_dims = [self.calcLayerDim(i) for i in range(self.cl+1)]
        self.encoder_series= nn.ModuleList([nn.Linear(i1,i2) for i1,i2 in zip(layer_dims[:3],layer_dims[1:])])
        self.decoder_series = nn.ModuleList([nn.Linear(i1,i2) for i1,i2 in zip(layer_dims[::-1][:3],layer_dims[::-1][1:])])

    def calcLayerDim(self,layer_index):
        # https://www.desmos.com/calculator/hxtsqbd1oc
        layer_dim = self.ld + (self.input_dim-self.ld)*((layer_index-self.cl)/self.cl)**self.exponent
        return int(layer_dim)

    def forward(self, x):

        # encode
        x = self.code(x,encode=True)
            
        # decode
        x = self.code(x,encode=False)

        return x
    
    def code(self,x,encode=True):
        
        series = self.encoder_series if encode else self.decoder_series
        
        for layer in series:
            x = layer(x)
            # x = F.relu(x)
        
        return x
    
    @staticmethod
    def name():
        return 'autoencoder'
    
def loadModelParams(model,file_name):
    path = f'./model_params/{file_name}.pt'
    model.load_state_dict(torch.load(path))

    return model