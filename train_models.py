from model import Net
from utils.data_utils import IDMT,MVD
from utils.training_utils import Trainer
from utils.evaluation_utils import Evaluator

def main():

    # idmt = IDMT()
    # trainloader = idmt.constructDataLoader()

    mvd = MVD()
    trainloader = mvd.constructDataLoader()

    print(len(trainloader))

    # output sometimes collapses when lr >= 10e-4

    # !! BE VERY CONSCIOUS OF LEARNING RATE !!
    # can either stop learning completely if too small or collapse it (divergence) if too large
    trainer = Trainer(Net(log=True),log=True,lr=10e-5)

    losses = trainer.training_epoch(epochs=3,trainloader=trainloader)

    print(losses[0],losses[-1])

    evaluator = Evaluator(trainer.model)
    accuracy = evaluator.evaluate(trainloader)
    print(f'Accuracy: {accuracy}')

if __name__ == '__main__':

    main()