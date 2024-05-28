from utils.sql_utils import DB
from utils.data_utils import Dataset, IDMT, MVD
import pandas as pd
from sklearn.datasets import load_iris

def main():
    data = load_iris()
    print(data)

if __name__ == '__main__':
    main()