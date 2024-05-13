# from model import Net

from models.nn import Shallow, Deep
from utils.data_utils import IDMT,MVD
from utils.grid_search_utils import GridSearch

def main(dir):

    idmt = IDMT()

    epochs = 5

    # models = [Shallow]
    models = [Shallow,Deep]
    learning_rates = [10**-i for i in range(5)]

    # trainloader
    feature_size,train_data,test_data = idmt.constructDataLoader(feature_set_type='statistical')

    gs = GridSearch(dir,epochs,train_data,test_data)

    gs.gridSearch(models,learning_rates)


if __name__ == '__main__':

    dir = '1'

    # create_folder(f'model_params/{dir}')

    main(dir)