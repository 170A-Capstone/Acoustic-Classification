
from utils.training_utils import Trainer
import time

class GridSearch():
    def __init__(self,feature_size,epochs,train_data,test_data,log = False) -> None:
        
        self.log = log

        self.feature_size = feature_size
        self.epochs = epochs
        self.train_data = train_data
        self.test_data = test_data
        
        if self.log:
            print('[Grid Search]: Grid Search initialized')

    def gridSearch(self,models,learning_rates,momenta):

        data = {}

        for model in models:
            
            model_data = {}

            for lr in learning_rates:

                lr_data = {}

                for momentum in momenta:

                    print(f'[Grid Search] Model: {model.name()} | LR: {lr} | Momentum: {momentum}')

                    model_instance = model(input_dim=self.feature_size,output_dim=4,log=False)

                    trainer = Trainer(model_instance,lr=lr,momentum=momentum,log=False)

                    accuracies = trainer.training_epoch(epochs=self.epochs,train_loader=self.train_data,val_loader=self.test_data)

                    momentum_data = {'accuracies':accuracies}
                            # 'loss':loss}
                    
                    lr_data[str(momentum)] = momentum_data

                model_data[str(lr)] = lr_data

            data[f'{model.name()}'] = model_data
            
        return data
    
    def ae_gridSearch(self,ae_class,learning_rates,coding_layers,latent_dim):

        print('[AE Grid Search] Beginning Grid Search')

        data = {}

        for lr in learning_rates:

            lr_data = {}

            for cl in coding_layers:

                cl_data = {}

                for ld in latent_dim:

                    a = time.time()

                    model_instance = ae_class(input_dim=self.feature_size,
                                              cl=cl,
                                              ld=ld)
                    
                    # print(model_instance.encoder_series[0])

                    # return

                    trainer = Trainer(model_instance,metric='loss',lr=lr,momentum=.9)

                    losses = trainer.training_epoch(epochs=self.epochs,train_loader=self.train_data,val_loader=self.test_data)

                    b = time.time()

                    print(f'[AE Grid Search] LR: {lr} | Coding Layers: {cl} | Latent Dimensionality: {ld} ({b-a:.2f}s)')

                    ld_data = {'losses':str(losses)}
                    
                    cl_data[str(ld)] = ld_data

                lr_data[str(cl)] = cl_data

            data[str(lr)] = lr_data
            
        return data

