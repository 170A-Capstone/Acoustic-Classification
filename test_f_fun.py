from utils.sql_utils import DB

def main():
    db = DB()
    df = db.downloadDF('MVD_features')
    print(df.shape)
    print(df.head())

if __name__ == '__main__':
    main()