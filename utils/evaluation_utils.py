import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

class Evaluator:
    """Utilities for evaluating accuracy of models
    """

    def __init__(self, model, model_path, data_loader, device='cpu'):
        self.model = model
        self.model.to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

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
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # forward pass -> probability of predicting each class
                outputs = self.model(inputs)

                # gets index of highest probability in output vector
                # index of highest probability = predicted label
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                iterations += 1

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
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                y_true.extend(labels.tolist())
                y_pred.extend(predicted.tolist())

        return confusion_matrix(y_true, y_pred)