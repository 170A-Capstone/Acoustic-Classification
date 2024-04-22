from utils.sql_utils import DB
from utils.data_utils import IDMT, MVD

def main():
    """Upload IDMT data to database
    """
    
    db = DB(log=True)
    idmt = IDMT(db, log=True)

    idmt.uploadFeatureDF(table_name='idmt_features')
    idmt.uploadAudioDF(table_name='idmt_audio')
    #idmt.audio_features_DF(table_name='idmt_audio_features')

    mvd = MVD(db, log=True)

    mvd.uploadFeatureDF(table_name='mvd_features')
#    mvd.audio_features_DF(table_name='mvd_audio_features')

    """Upload MVD data to database
    """
    """
    dbMVD = DB(log=True)
    mvd = MVD(log=True)

    pathsmvd = mvd.getFilePaths()#[:5]
    feature_dfmvd = mvd.getFeatureDF(pathsmvd)

    dbMVD.uploadDF(df=feature_dfmvd,table_name='mvd_metadata')

    audio_dfmvd = mvd.extractAudioDF(pathsmvd)

    dbMVD.uploadDF(df=audio_dfmvd,table_name='mvd_audio_left')
"""
    
if __name__ == '__main__':
    main()