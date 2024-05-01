from utils.data_utils import IDMT,MVD

def main():

    # idmt = IDMT()
    # idmt.transformSignals(transform='statistical')

    mvd = MVD()
    mvd.transformSignals(transform='statistical')

if __name__ == '__main__':

    main()