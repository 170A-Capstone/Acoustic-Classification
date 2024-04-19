from utils.sql_utils import DB
import numpy as np

import time

def decodeBLOb(blob):
    return np.frombuffer(blob, dtype=np.float32)

def constructDataLoader(db):
    query_str = f'''
        SELECT features.class,audio.audio 
        FROM idmt_metadata AS features 
        LEFT JOIN idmt_audio_left AS audio 
        ON features.index = audio.index 
        '''

    a = time.time()

    db.cursor.execute(query_str)
    data = db.cursor.fetchall()

    b = time.time()

    # decode audio waveforms
    data = [(class_label,decodeBLOb(blob)) for class_label,blob in data]

    c = time.time()

    print('fetch:',b-a)
    print('decode:',c-b)
    print('total:',c-a)

def main():

    db = DB(log=True)

    # 98 seconds
    # audio_waveforms = db.downloadBLObs('idmt_audio_left')

    # 92 seconds
    data_loader = constructDataLoader(db)

    print(len(data_loader))
    print(len(data_loader[0][1]))
    print(data_loader[0])


if __name__ == '__main__':

    main()