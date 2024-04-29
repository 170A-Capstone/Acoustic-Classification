import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """ResNet model

    outputs 4 numbers, corresponding to a vector of probabilities for each possible class (car, truck, etc)
    """

    def __init__(self,log = False):

        self.log = True

        super(Net, self).__init__()
        
        # fft, .process
        self.fc = nn.Linear(480, 4)

        # statistical features
        self.fc = nn.Linear(13, 4)

        if self.log:
            print('[Model]: Model initialized')
        

    def forward(self, x):
        x = self.fc(x)
        return x