import os, sys, torch

from utils.file_utils import *
from utils.training_utils import Trainer
from model import Net

import utils.preprocessing_utils as preproc
from utils.data_utils import IDMT

def storeParameters():
    
    # create folder to store loss and model parameters for particular training run
    create_folder(folder_path)
    
    # save loss
    path = f"./models/{test_label}/loss_output.csv"
    store_csv(loss_output,regularizer_labels,path)

    # save parameters
    for model_package in model_components:
        model_parameters = model_package["model"].state_dict()
        model_path = f'./models/{test_label}/{model_package["label"]}.pt'

        torch.save(model_parameters, model_path)

def readArguments(args):
    if len(args) != 3:
        print("Usage: python train_models.py <test label> <lambda reg>")
        raise SyntaxError
    else:
        test_label = sys.argv[1]

        # get lambda_reg from arguments
        try:
            lambda_reg = float(sys.argv[2])
        except:
            raise TypeError
    
    # if folder already exists, break
    folder_path = f"./models/{test_label}"
    if check_folder_exists(folder_path):
        raise FileExistsError
    
    return lambda_reg,test_label

import numpy as np

def main():
    net = Net(log=True)
    trainer = Trainer(net,log=True)

    idmt = IDMT(log=True,compression_factor=None)
    paths = idmt.getFilePaths()

    # construct training dataset
    trainloader = []
    for path in paths[:5]:
        audio = idmt.extractAudio(path)
        fft,compressed_fft = preproc.process(audio)

        label_embedding = idmt.extractLabelEmbedding(path)

        trainloader.append((compressed_fft,label_embedding))

    # print(trainloader)

    losses = trainer.training_epoch(epochs=1,trainloader=trainloader)

    print(losses)

    # index = 2
    # path = paths[index]
    # audio = idmt.extractAudio(path)
    # label_embedding = idmt.extractLabelEmbedding(path)

    # fft,compressed_fft = preproc.process(audio)

    # loss = trainer.training_loop(compressed_fft,label_embedding)
    # print(loss)

if __name__ == '__main__':

    # lambda_reg,test_label = readArguments(sys.argv)
    # main(lambda_reg,test_label)

    main()