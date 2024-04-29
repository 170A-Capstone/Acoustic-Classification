from utils.sql_utils import DB
from utils.data_utils import IDMT,MVD

def main():

    db = DB()

    # idmt = IDMT(db,log=True)
    # idmt.transformSignals(transform='statistical')

    mvd = MVD(db,log=True)
    mvd.transformSignals(transform='statistical')

if __name__ == '__main__':

    main()