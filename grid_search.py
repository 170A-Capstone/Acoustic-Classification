# from model import Net

from models.nn import Shallow, Deep, AutoEncoder, Deep2
from utils.data_utils import IDMT,MVD
from utils.grid_search_utils import GridSearch
from utils.file_utils import create_folder,saveJson

def main(dir):

    epochs = 10

    # models = [AutoEncoder]
    models = [Shallow,Deep,AutoEncoder,Deep2]

    # learning_rates = [10e-6]
    learning_rates = [10**-i for i in range(1,7)]

    momenta = [.9]
    # momenta = [n/100 for n in range(70,105,10)]

    # trainloader
    feature_size,train_data,test_data = MVD().constructDataLoader(feature_set_type='statistical-PCA')
    
    gs = GridSearch(feature_size,epochs,train_data,test_data)

    data = gs.gridSearch(models,learning_rates,momenta)

    saveJson(file_path=f'./model_params/{dir}/results.json',data=data)


if __name__ == '__main__':

    dir = 'mvd-10'

    create_folder(f'model_params/{dir}')

    main(dir)