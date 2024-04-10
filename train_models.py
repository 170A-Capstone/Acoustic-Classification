import os, sys, torch

from utils.file_utils import *
from utils.training_utils import Trainer
from model import Net

import utils.preprocessing_utils as preproc
from utils.data_utils import IDMT

def main():
    net = Net(log=True)

    # output sometimes collapses when lr >= 10e-4
    trainer = Trainer(net,log=True,lr=10e-6)

    idmt = IDMT(log=True,compression_factor=None)
    paths = idmt.getFilePaths()

    trainloader = idmt.constructDataLoader(paths[:10])
    losses = trainer.training_epoch(epochs=2,trainloader=trainloader)

    print(losses)

if __name__ == '__main__':

    # lambda_reg,test_label = readArguments(sys.argv)
    # main(lambda_reg,test_label)

    main()