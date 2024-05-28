from utils.sql_utils import DB
from utils.data_utils import Dataset, IDMT, MVD
import pandas as pd
from sklearn.datasets import load_iris

def main():
    db = DB()
    idmt_df = db.downloadDF('IDMT_statistical_features')

    print(idmt_df.shape)

if __name__ == '__main__':
    main()