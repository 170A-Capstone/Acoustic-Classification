# from model import Net
from sklearn.model_selection import GridSearchCV
from models.nn import Shallow, Deep
from utils.data_utils import IDMT,MVD
from utils.training_utils import Trainer

def main():

    idmt = IDMT()
    
    epochs = 10

    features = ['ambient']
    # models = [Deep]
    models = [Shallow,Deep]
    # param_2 = ['a','b']

    for feature_set_type in features:

        # trainloader
        feature_size,trainloader = idmt.constructDataLoader(feature_set_type=feature_set_type)

        for model in models:

            # trainer = None
            # if model == 'shallow':
            #     trainer = Trainer(Shallow(input_dim=feature_size))

            print(f'[Grid Search] Feature: {feature_set_type} | Model: {model}')

            trainer = Trainer(model(input_dim=feature_size))

            trainer.training_epoch(epochs=epochs,trainloader=trainloader)

            # for p in param_2:
            #     print(f'[Grid Search] Feature: {feature_set_type} | Model: {model} | Param: {p}')

                # train model

                # store model params + loss

    pass

if __name__ == '__main__':

    main()