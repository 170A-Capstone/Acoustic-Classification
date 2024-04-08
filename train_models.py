import os, sys, torch

from utils.file_utils import *
from utils.training_utils import Trainer

def main(lambda_reg,test_label):
    """Train models on four significant regularizers—the last of which is novel—on CIFAR-10 dataset and store model parameters in training context directory inside ./models/ folder

    Args:
        lambda_reg (float): modulation value to scale the impact of regularization terms
        test_label (string): name of training context folder that will be created to store trained model parameters
    """

    loss_output = []

    print('= [Training]')
    for epoch in range(10):

        print('== [Epoch: %s]' % epoch)

        # time epoch
        start_time = time.time()

        for i, (inputs,labels) in enumerate(trainloader, 0):

            # limit training time for debugging purposes
            # if i > 5:
            #     break

            loss_output_for_epoch = []

            # iterate through all regularizer functions
            for model_package in model_components:

                # run a single training loop
                loss = training_loop(model_package,lambda_reg,inputs,labels)

                loss_output_for_epoch.append(loss)
        
        loss_output.append(loss_output_for_epoch)

        end_time = time.time()
        print(end_time - start_time)

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

if __name__ == '__main__':

    if len(sys.argv) != 3:
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
    
    main(lambda_reg,test_label)