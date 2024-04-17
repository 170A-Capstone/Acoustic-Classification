from model import Net
from utils.data_utils import IDMT
from utils.training_utils import Trainer
from utils.evaluation_utils import Evaluator

def main():
    net = Net(log=True)

    # output sometimes collapses when lr >= 10e-4
    trainer = Trainer(net,log=True,lr=10e-6)

    idmt = IDMT(log=True,compression_factor=None)
    paths = idmt.getFilePaths()

    trainloader = idmt.constructDataLoader(paths[:10])
    losses = trainer.training_epoch(epochs=10,trainloader=trainloader)

    print(losses)

    evaluator = Evaluator(trainer.model)
    accuracy = evaluator.evaluate(trainloader[:1000])
    print(f'accuracy: {accuracy}')

if __name__ == '__main__':

    main()