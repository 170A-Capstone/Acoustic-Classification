from utils.sql_utils import DB
from utils.data_utils import IDMT

def main():
    """Upload IDMT data to database
    """
    
    db = DB(log=True)
    idmt = IDMT(log=True)

    paths = idmt.getFilePaths()[:5]
    feature_df = idmt.getFeatureDF(paths)

    db.uploadDF(df=feature_df,table_name='idmt_metadata')

    audio_df = idmt.extractAudioDF(paths)

    db.uploadDF(df=audio_df,table_name='idmt_audio_left')

if __name__ == '__main__':
    main()