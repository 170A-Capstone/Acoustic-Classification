from utils.sql_utils import DB

def main():
    db = DB(log=True)
    
    df = db.downloadDF('IDMT_statistical_features')

    print(df.head())

if __name__ == '__main__':
    main()