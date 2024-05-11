from utils.sql_utils import DB

def main():
    db = DB()
    df = db.downloadDF('IDMT_features')
    y = df['class']          
    print(y.unique())

if __name__ == '__main__':
    main()