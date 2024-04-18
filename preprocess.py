from utils.sql_utils import DB
from utils.data_utils import IDMT, MVD

def main():
    """Upload IDMT data to database
    """
    
    db = DB(log=True)
    idmt = IDMT(log=True)

    paths = idmt.getFilePaths()#[:5]
    feature_df = idmt.getFeatureDF(paths)

    db.uploadDF(df=feature_df,table_name='idmt_metadata')

    audio_df = idmt.extractAudioDF(paths)

    db.uploadDF(df=audio_df,table_name='idmt_audio_left')

    """Upload MVD data to database
    """
    dbMVD = DB(log=True)
    mvd = MVD(log=True)

    pathsmvd = mvd.getFilePaths()#[:5]
    feature_dfmvd = mvd.getFeatureDF(pathsmvd)

    dbMVD.uploadDF(df=feature_dfmvd,table_name='mvd_metadata')

    audio_dfmvd = mvd.extractAudioDF(pathsmvd)

    dbMVD.uploadDF(df=audio_dfmvd,table_name='mvd_audio_left')

    
if __name__ == '__main__':
    main()