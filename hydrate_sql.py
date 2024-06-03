from utils.sql_utils import DB
from utils.data_utils import IDMT,MVD

def main():
    """Upload data and signals to database
    """

    db = DB(log=True)
    
    idmt = IDMT(db,log=True)
    idmt.uploadFeatures()
    idmt.uploadSignals()
    idmt.transformSignals('statistical')
    idmt.transformSignals('librosa')
    
    mvd = MVD(db,log=True)
    mvd.uploadFeatures()
    mvd.uploadSignals()
    mvd.transformSignals('statistical')
    mvd.transformSignals('librosa')

if __name__ == '__main__':
    main()