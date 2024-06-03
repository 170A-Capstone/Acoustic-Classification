from utils.data_utils import IDMT,MVD,IDMT_BG

def main():

    # idmt_bg = IDMT_BG()
    # idmt_bg.transformSignals(transform='harmonic')

    # idmt = IDMT()
    # idmt.transformSignals(transform='statistical')

    mvd = MVD()
    mvd.transformSignals(transform='statistical')

if __name__ == '__main__':

    main()