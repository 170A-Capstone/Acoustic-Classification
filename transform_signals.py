from utils.data_utils import IDMT,MVD,IDMT_BG
from utils.encoding_utils import IDMT_Encode

def main():

    # idmt_bg = IDMT_BG()
    # idmt_bg.transformSignals(transform='harmonic')

    idmt = IDMT_Encode(params_path='idmt-cae',coding_layers=2,latent_dim=8)
    idmt.transformSignals(transform='encode')

    # idmt = IDMT()
    # idmt.downsampleSignals()

    # mvd = MVD()
    # mvd.downsampleSignals()
    # mvd.transformSignals(transform='statistical')

if __name__ == '__main__':

    main()