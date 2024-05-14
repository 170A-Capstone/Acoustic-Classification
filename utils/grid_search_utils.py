
from utils.training_utils import Trainer
from utils.evaluation_utils import Evaluator

class GridSearch():
    def __init__(self,epochs,train_data,test_data,log = False) -> None:
        
        self.log = log

        self.epochs = epochs
        self.train_data = train_data
        self.test_data = test_data
        
        if self.log:
            print('[Grid Search]: Grid Search initialized')

    def trainAndEvaluateModel(self,model,lr,momentum):

        trainer = Trainer(model,lr=lr,momentum=momentum,log=False)

        loss = trainer.training_epoch(epochs=self.epochs,trainloader=self.train_data)
        
        evaluator = Evaluator(model,log=False)
        accuracy = evaluator.evaluate(data_loader=self.test_data)

        return accuracy,loss
    
    def gridSearch(self,models,learning_rates,momenta):

        data = {}

        for model in models:
            
            model_data = {}

            for lr in learning_rates:

                lr_data = {}

                for momentum in momenta:

                    print(f'[Grid Search] Model: {model.name()} | LR: {lr} | Momentum: {momentum}')

                    model_instance = model(input_dim=6,output_dim=4,log=False)

                    accuracy,loss = self.trainAndEvaluateModel(model_instance,lr,momentum)
                    # print(accuracy)

                    momentum_data = {'accuracy':accuracy,
                            'loss':loss}
                    
                    lr_data[str(momentum)] = momentum_data

                model_data[str(lr)] = lr_data

            data[f'{model.name()}'] = model_data
            
        return data

