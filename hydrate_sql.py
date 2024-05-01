from utils.data_utils import IDMT,MVD

def main():
    """Upload data and signals to database
    """

    idmt = IDMT()
    idmt.uploadFeatures()
    idmt.uploadSignals()
    
    mvd = MVD()
    mvd.uploadFeatures()
    mvd.uploadSignals()

if __name__ == '__main__':
    main()