
from utils.training_utils import Trainer
from utils.evaluation_utils import Evaluator
from utils.file_utils import saveJson

class GridSearch():
    def __init__(self,dir,epochs,train_data,test_data,log = False) -> None:
        
        self.log = log

        self.dir = dir
        self.epochs = epochs
        self.train_data = train_data
        self.test_data = test_data
        
        if self.log:
            print('[Grid Search]: Grid Search initialized')

    def trainAndEvaluateModel(self,model,lr):

        trainer = Trainer(model,lr=lr,log=False)

        loss = trainer.training_epoch(epochs=self.epochs,trainloader=self.train_data)
        
        evaluator = Evaluator(model,log=False)
        accuracy = evaluator.evaluate(data_loader=self.test_data)

        return accuracy,loss
    
    def gridSearch(self,models,learning_rates):

        data = {}

        for model in models:
            
            model_data = {}

            for lr in learning_rates:

                print(f'[Grid Search] Model: {model.name()} | LR: {lr}')

                model_instance = model(input_dim=13,output_dim=4,log=False)

                accuracy,loss = self.trainAndEvaluateModel(model_instance,lr)

                lr_data = {'accuracy':accuracy,
                           'loss':loss}

                model_data[str(lr)] = lr_data

            data[f'{model.name()}'] = model_data
            
        saveJson(file_path=f'./model_params/{self.dir}/results.json',data=data)


