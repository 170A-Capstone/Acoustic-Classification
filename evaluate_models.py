import torch, sys

from utils.evaluation_utils import Evaluator
from utils.file_utils import *

def main(folder_path):
    """Evaluates trained models stored in the given training context

    Args:
        folder_path (string): name of the folder inside ./models/ corresponding to the training context to be tested
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    evaluator = Evaluator(Model(), file_path, testloader, device)
    accuracy,latency = evaluator.evaluate()

    print("accuracy",accuracy)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python train_models.py <test label>")
        raise SyntaxError
    else:
        test_label = sys.argv[1]

    # if folder DNE, break
    folder_path = f"./models/{test_label}"
    if not check_folder_exists(folder_path):
        raise FileNotFoundError

    main(folder_path)