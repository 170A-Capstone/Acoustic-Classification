from utils.sql_utils import DB
from utils.data_utils import IDMT

def main():
    """Upload IDMT data to database
    """
    
    db = DB(log=True)
    idmt = IDMT(log=True,compression_factor=None)

    paths = idmt.getFilePaths()

    # upload features
    feature_df = idmt.getFeatureDF(paths)
    db.uploadDF(df=feature_df,table_name='idmt_metadata')

    # upload audio

    # 78 seconds
    audio_waveforms = [idmt.extractAudio(path) for path in paths]

    # 117 seconds
    db.uploadBLObs(audio_waveforms,'idmt_audio_left')

if __name__ == '__main__':
    main()