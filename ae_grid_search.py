# from model import Net

from models.nn import ConvolutionalAutoEncoder
from utils.data_utils import IDMT,MVD
from utils.grid_search_utils import GridSearch
from utils.file_utils import create_folder,saveJson

def main(dir):

    epochs = 5

    # learning_rates = [10e-1]
    learning_rates = [10**-i for i in range(1,3)]
    # learning_rates = [.001,.005,.01,.05]

    # minimum of 1
    coding_layers = [3]
    # coding_layers = list(range(1,3))

    # latent_dim = [6]
    latent_dim = list(range(6,12,2))

    # trainloader
    # feature_size,train_data,test_data = IDMT().constructDataLoader(feature_set_type='raw')
    feature_size,train_data,test_data = IDMT().construct_ae_DataLoader()

    gs = GridSearch(feature_size,epochs,train_data,test_data)

    data = gs.ae_gridSearch(ConvolutionalAutoEncoder,learning_rates,coding_layers,latent_dim)

    # print(data)

    saveJson(file_path=f'./model_params/{dir}/results.json',data=data)


if __name__ == '__main__':

    dir = 'cae-4-3'

    create_folder(f'model_params/{dir}')

    main(dir)