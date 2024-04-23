import utils.signal_processing_utils as sp
from utils.data_utils import IDMT
from utils.sql_utils import DB
import pandas as pd

def main():

    idmt = IDMT(DB(),log=True)
    idmt.transformSignals(transform='statistical')

if __name__ == '__main__':

    main()