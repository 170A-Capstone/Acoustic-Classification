from utils.sql_utils import DB
from utils.data_utils import Dataset, IDMT, MVD
import pandas as pd

def main():
    db = DB()
    idmt_df = db.downloadDF('IDMT_features')
    mvd_df = db.downloadDF('MVD_features')

    df = pd.concat([idmt_df, mvd_df], ignore_index=True)
    print(df.shape)
    print(df.head())

if __name__ == '__main__':
    main()