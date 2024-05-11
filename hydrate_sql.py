from utils.data_utils import IDMT,MVD,IDMT_BG

def main():
    """Upload data and signals to database
    """

    idmt_bg = IDMT_BG()
    idmt_bg.uploadFeatures()
    idmt_bg.uploadSignals()
    
    idmt = IDMT()
    idmt.uploadFeatures()
    idmt.uploadSignals()
    
    mvd = MVD()
    mvd.uploadFeatures()
    mvd.uploadSignals()

if __name__ == '__main__':
    main()