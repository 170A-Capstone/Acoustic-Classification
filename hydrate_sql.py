from utils.sql_utils import DB
from utils.data_utils import IDMT

def main():
    """Upload IDMT data to database
    """
    
    db = DB(log=False)
    idmt = IDMT(db,log=True)
    idmt.uploadFeatureDF()
    idmt.uploadSignalsDF()

if __name__ == '__main__':
    main()