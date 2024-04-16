import psycopg2
import pymysql
import mysql.connector
import pandas as pd
from sqlalchemy import create_engine

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

        self.cursor = self.connectCursor()
        self.engine = self.connectEngine()

    def connectCursor(self):
        conn = mysql.connector.connect(host='localhost',
                            #database='170',
                            user='root',
                            password='13683819895Rjy',
                            port=3306)
        cursor = conn.cursor()
        return cursor

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