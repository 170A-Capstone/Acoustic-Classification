from models.nn import ConvolutionalAutoEncoder
from utils.data_utils import IDMT
from utils.training_utils import Trainer

def main():

    coding_layers = 2
    latent_dim = 8

    learning_rate = 10e-1
    epochs = 5

    # trainloader
    feature_size,train_data,test_data = IDMT().construct_ae_DataLoader()

    print(feature_size)

    return

    model_instance = ConvolutionalAutoEncoder(input_dim=feature_size,
                                cl=coding_layers,
                                ld=latent_dim)
    
    trainer = Trainer(model_instance,metric='loss',lr=learning_rate,momentum=.9)

    losses = trainer.training_epoch(epochs=epochs,train_loader=train_data,val_loader=test_data)

    print(losses)

    trainer.storeParams(file_name='cae')

if __name__ == '__main__':

    main()