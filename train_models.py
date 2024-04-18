import os, sys, torch

from utils.file_utils import *
from utils.training_utils import Trainer
from model import Net
from utils.data_utils import IDMT
from utils.training_utils import Trainer
from utils.evaluation_utils import Evaluator

def main():
    net = Net(log=True)

    # output sometimes collapses when lr >= 10e-4
    trainer = Trainer(net,log=True,lr=10e-6)

    idmt = IDMT(log=True,compression_factor=None)
    paths = idmt.getFilePaths()

    trainloader = idmt.constructDataLoader(paths[:10])
    testloader = idmt.constructDataLoader(paths[11:20])
    losses = trainer.training_epoch(epochs=2,trainloader=trainloader)

    print(losses)

    model_path= "/Users/cecima/Desktop/170B-Proj/Acoustic-Classification/state_dict_model.pt"
    torch.save(trainer.model.state_dict(), model_path)

    evaluator_test = Evaluator(trainer, model_path, testloader)
    print(evaluator_test.evaluate())
    print(evaluator_test.get_confusion_matrix())


# Directly create the evaluation class here
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix

class Evaluator:
    """Utilities for evaluating accuracy of models
    """

    def __init__(self, model, model_path, data_loader, device='cpu'):
        self.model = model
        self.model.model.to(device)
        self.model.model.load_state_dict(torch.load(model_path))
        self.model.model.eval()

        self.data_loader = data_loader
        self.device = device

    def evaluate(self) -> float:
        """Measures accuracy of a model at the time of inference

        Returns:
            float: model accuracy [0,1]
        """
        correct = 0
        total = 0

        iterations = 0

        with torch.no_grad():
            for inputs, labels in self.data_loader:
                inputs = torch.Tensor(inputs)
                # what is the purpose of this?
                # inputs, labels = inputs.to(self.device), labels.to(self.device)

                # forward pass -> probability of predicting each class
                outputs = self.model.model(inputs)

                # gets index of highest probability in output vector
                # index of highest probability = predicted label
                value,predicted_class = torch.max(outputs.data,0)
                predicted_class = predicted_class.item()
                
                actual_class = labels.index(1)

                if predicted_class == actual_class:
                    correct += 1
                total += 1

        accuracy = correct / total

        return accuracy
    
    def correlation_matrix(self):
        """Calculate and plot the correlation matrix for features in the data.

        Returns:
            pd.DataFrame: Correlation matrix of the features.
        """
        # Assuming data_loader can be iterated to yield a DataFrame
        frames = [data for data, _ in self.data_loader]
        full_data = pd.concat(frames)
        corr_matrix = full_data.corr()

        # Plotting the correlation matrix
        plt.figure(figsize=(10, 8))  # Set the figure size as needed
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                    cbar=True, square=True, linewidths=.5)
        plt.title('Correlation Matrix Heatmap')
        plt.show()

        return corr_matrix
    
    def get_confusion_matrix(self):
        """Compute the confusion matrix for model predictions.

        Returns:
            np.array: Confusion matrix
        """
        y_pred = []
        y_true = []

        with torch.no_grad():
            for inputs, labels in self.data_loader:
                inputs = torch.Tensor(inputs)
                outputs = self.model.model(inputs)
                value, predicted = torch.max(outputs.data, 0)

                y_true.append(labels.index(1))
                y_pred.append(predicted.item())

        return confusion_matrix(y_true, y_pred, labels=[0,1,2,3])



if __name__ == '__main__':

    main()