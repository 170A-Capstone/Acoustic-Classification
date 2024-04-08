import torch

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