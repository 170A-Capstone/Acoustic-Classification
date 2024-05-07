from models.nn import Shallow,loadModelParams
from utils.data_utils import IDMT,MVD,IDMT_BG
from utils.evaluation_utils import Evaluator

def main():

    idmt_bg = IDMT_BG()
    feature_size,trainloader = idmt_bg.constructDataLoader()

    model = Shallow(input_dim=feature_size,output_dim=1)
    model = loadModelParams(model,file_name='signal_detection_from_harmonics')

    evaluator = Evaluator(model)
    accuracy = evaluator.evaluate(trainloader[:1000])
    print(f'Accuracy: {accuracy}')

if __name__ == '__main__':

    main()