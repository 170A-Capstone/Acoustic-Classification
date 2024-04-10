import torch
import torch.optim as optim
import torch.nn as nn

class Trainer():
    def __init__(self,model,lr=0.001, momentum=0.9,log = False) -> None:
        
        self.log = log
        self.model = model

        # baseline loss function
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        if self.log:
            print('[Trainer]: Trainer initialized')

    def training_loop(self,inputs,labels) -> "loss":

        inputs = torch.Tensor(inputs)
        labels = torch.Tensor(labels)

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward pass
        outputs = self.model(inputs)

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

        for epoch in range(epochs):

            print(f'[Trainer]: Epoch {epoch}')

            for i, (inputs,labels) in enumerate(trainloader, 0):

                # limit training time for debugging purposes
                # if i > 5:
                #     break
                
                loss = self.training_loop(inputs,labels)
            
                losses.append(loss)

        return losses