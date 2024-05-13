import torch
import torch.optim as optim
import torch.nn as nn
import time

class Trainer():
    def __init__(self,model,lr=0.001, momentum=0.9,log = False) -> None:
        
        self.log = log
        self.model = model

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # net.to(device)

        # baseline loss function
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.MSELoss()

        print(lr)

        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

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
    
    def training_epoch(self,epochs,trainloader):
        losses = []
        loss = 0

        if self.log:
            print(f'[Trainer]: Training on {len(trainloader)} data points')

        for epoch in range(epochs):

            if self.log:
                a = time.time()

            for (inputs,labels) in trainloader:

                loss = self.training_loop(inputs,labels)
            
            losses.append(loss)
            
            if self.log:
                b = time.time()
                print(f'[Trainer]: Completed Epoch {epoch} ({b-a:.2f}s)')


        return losses

    def storeParams(self,file_name):
        model_parameters = self.model.state_dict()

        path = f'./model_params/{file_name}.pt'
        torch.save(model_parameters, path)

        if self.log:
            print(f'[Trainer]: Model Parameters stored in {path}.')