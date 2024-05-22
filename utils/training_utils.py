import torch
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np

class Trainer():
    def __init__(self,model,lr=0.001, momentum=0.9,metric='accuracy',log = False) -> None:
        
        self.log = log
        self.model = model
        self.metric = metric

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # net.to(device)

        # baseline loss function
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.MSELoss()

        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        if self.log:
            print('[Trainer]: Trainer initialized')

        # if self.log:
        #     print('[Trainer]: Parameter summary:')
        #     print(summarizeWeights(self.model))

    def training_loop(self,inputs,labels) -> "loss":

        inputs = torch.Tensor(inputs)
        labels = torch.Tensor(labels)

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward pass
        outputs = self.model(inputs)
        # print(outputs)

        # calculate baseline loss + modulated regularization value
        loss = self.criterion(outputs, labels)

        # calculate gradients with respect to loss
        loss.backward()

        # apply gradients to parameters
        self.optimizer.step()

        # return loss value for analysis
        return loss.item()
    
    def training_epoch(self,epochs,train_loader,val_loader):
        metrics = []

        if self.log:
            print(f'[Trainer]: Training on {len(train_loader)} data points')

        # evaluate metric with no training
        self.model.eval()
        metric = self.evaluate(val_loader)
        metrics.append(metric)

        for epoch in range(epochs):

            if self.log:
                a = time.time()

            # train
            self.model.train()
            for (inputs,labels) in train_loader:
                self.training_loop(inputs,labels)

            # evaluate
            self.model.eval()
            metric = self.evaluate(val_loader)
            metrics.append(metric)
            
            if self.log:
                b = time.time()
                print(f'[Trainer]: Completed Epoch {epoch} ({b-a:.2f}s)')

        return metrics
    
    def evaluate(self,data_loader):
        if self.metric == 'accuracy':
            return self.evaluateAccuracy(data_loader)
        if self.metric == 'loss':
            return self.evaluateLoss(data_loader)
    
    def evaluateAccuracy(self,data_loader) -> float:
        """Measures accuracy of a model at the time of inference

        Returns:
            float: model accuracy [0,1]
        """
        correct = 0

        # if self.log:
        #     a = time.time()

        with torch.no_grad():
            for inputs, labels in data_loader:

                inputs = torch.Tensor(inputs)
                # print(inputs)

                # what is the purpose of this?
                # inputs, labels = inputs.to(self.device), labels.to(self.device)

                # forward pass -> probability of predicting each class
                outputs = self.model(inputs)
                # print(outputs)

                # gets index of highest probability in output vector
                # index of highest probability = predicted label
                value,predicted_class = torch.max(outputs.data,0)
                predicted_class = predicted_class.item()

                actual_class = labels.index(1)

                # if predicted_class != 0:
                if predicted_class == actual_class:
                    correct += 1

        accuracy = correct / len(data_loader)

        # if self.log:
        #     b = time.time()
        #     print(f'[Evaluator]: Model Accuracy Evaluated ({b-a:.2f}s)')

        return accuracy

    def evaluateLoss(self,data_loader):

        losses = []

        with torch.no_grad():
            for inputs, labels in data_loader:
            
                inputs = torch.Tensor(inputs)
                labels = torch.Tensor(labels)
                

                # forward pass
                outputs = self.model(inputs)

                # calculate baseline loss + modulated regularization value
                loss = self.criterion(outputs, labels)

                losses.append(loss)

        return np.mean(losses)

    def storeParams(self,file_name):
        model_parameters = self.model.state_dict()

        path = f'./model_params/{file_name}.pt'
        torch.save(model_parameters, path)

        if self.log:
            print(f'[Trainer]: Model Parameters stored in {path}.')