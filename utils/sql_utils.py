import psycopg2
import pymysql
import mysql.connector
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

##import sql_credentials

class DB():
    """Utilities for interfacing with postgreSQL database

    Credentials are derived from file "sql_credentials.py"
        Example: 

        sql_credentials = {
            'host':'localhost',
            'dbname':'auralytics',
            'user':'postgres',
            'password':'password',
            'port':'5433'
        }

        sql_credentials = {
            'host':'MacBook-Pro-2.local',
            'dbname':'170',
            'user':'mysql',
            'password':'13683819895Rjy',
            'port':'3306'
        }
    """

    def __init__(self,log = False) -> None:
        self.log = log

        self.conn, self.cursor = self.connectCursor()
        self.engine = self.connectEngine()

    def connectCursor(self):
        conn = mysql.connector.connect(host='localhost',
                            database='project170',
                            user='root',
                            password='13683819895Rjy',
                            port=3306)
        cursor = conn.cursor()
        return conn,cursor

    def connectEngine(self):

        # construct connection string from sql credentials
        cnxn_str = 'mysql+pymysql://root:13683819895Rjy@localhost:3306/project170'
        cnxn_str = cnxn_str.format(host='localhost',
                            database='project170',
                            user='root',
                            password='13683819895Rjy',
                            port=3306,
                            )

        # establish connection
        engine = create_engine(cnxn_str)

        return engine
    
    def uploadDF(self,df,table_name):
        """Upload dataframe to database

        Args:
            df (pandas.DataFrame): dataframe containing data to be stored
            table_name (str): name of table display name
        """
        
        df.to_sql(table_name, con=self.engine, if_exists='replace')
        
        if self.log:
            print(f'[SQL]: Table "{table_name}" Uploaded')

    def downloadDF(self,table_name):
        """Query dataframe from database

        Args:
            table_name (str): name of table display name

        Returns:
            pandas.DataFrame: dataframe containing queried data
        """

        try:
            with self.engine.connect() as conn: 
                df = pd.read_sql(table_name, conn)
            return df
        except:
            print(f'[SQL]: ! Unable to download Table "{table_name}" !')

    def uploadBLObs(self,audio_waveforms,table_name):

        # create new table if necessary
        s = f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INT PRIMARY KEY,
                audio BLOB NOT NULL
            )
            '''

        self.cursor.execute(s)

        # !! COULD BE A HAZARD (inserting into non-committed table)
        # self.conn.commit()

        query_str = f'insert into {table_name}(index,audio) values (%s,%s)'

        # upload audio waveforms as BLObs with corresponding identifier (index)
        for index,audio_waveform in enumerate(audio_waveforms):

            # convert audio to BLOb
            blob = psycopg2.Binary(audio_waveform)

            # upload to database
            self.cursor.execute(query_str,(index,blob))

        # commit changes
        self.conn.commit()
                
        if self.log:
            print(f'[SQL]: BLObs Uploaded to Table "{table_name}"')


    def downloadBLObs(self,table_name):
        
        # fetch audio waveforms
        query_str = f'''
            SELECT * FROM {table_name}
            ORDER BY index ASC 
            '''

        self.cursor.execute(query_str)
        data = self.cursor.fetchall()

        # decode audio waveforms
        audio_waveforms = [np.frombuffer(bin, dtype=np.float32) for index,bin in data]

        if self.log:
            print(f'[SQL]: BLObs Downloaded from Table "{table_name}"')

        return audio_waveforms