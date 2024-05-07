# from model import Net
from models.nn import Shallow
from utils.data_utils import IDMT,MVD,IDMT_BG
from utils.training_utils import Trainer
from utils.evaluation_utils import Evaluator

def main():

    idmt_bg = IDMT_BG()
    feature_size,trainloader = idmt_bg.constructDataLoader()

    # idmt = IDMT()
    # trainloader = idmt.constructDataLoader()

    # mvd = MVD()
    # trainloader = mvd.constructDataLoader()

    model = Shallow(input_dim=feature_size,output_dim=1)

    # !! BE VERY CONSCIOUS OF LEARNING RATE !!
    # can either stop learning completely if too small or collapse it (divergence) if too large
    trainer = Trainer(model,log=True,lr=10e-5)

    losses = trainer.training_epoch(epochs=1,trainloader=trainloader)

    # print(losses[0],losses[-1])

    trainer.storeParams(file_name='signal_detection_from_harmonics')

    # evaluator = Evaluator(trainer.model)
    # accuracy = evaluator.evaluate(trainloader)
    # print(f'Accuracy: {accuracy}')

if __name__ == '__main__':

    main()