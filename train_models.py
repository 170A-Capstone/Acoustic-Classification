from model import Net
from utils.sql_utils import DB
from utils.data_utils import IDMT
from utils.training_utils import Trainer
from utils.evaluation_utils import Evaluator

def main():

    idmt = IDMT(DB(),log=True)
    trainloader = idmt.constructDataLoader()

    # output sometimes collapses when lr >= 10e-4
    trainer = Trainer(Net(log=True),log=True,lr=10e-6)

    losses = trainer.training_epoch(epochs=1,trainloader=trainloader)

    # print(losses)

    evaluator = Evaluator(trainer.model)
    accuracy = evaluator.evaluate(trainloader[-1000:])
    print(f'Accuracy: {accuracy}')

if __name__ == '__main__':

    main()