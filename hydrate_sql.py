from utils.sql_utils import DB
from utils.data_utils import IDMT,MVD

def main():
    """Upload data and signals to database
    """

    db = DB(log=False)
    
    idmt = IDMT(db,log=True)
    idmt.uploadFeatures()
    idmt.uploadSignals()
    
    mvd = MVD(db,log=True)
    mvd.uploadFeatures()
    mvd.uploadSignals()

if __name__ == '__main__':
    main()