from utils.sql_utils import DB
from utils.data_utils import Dataset, IDMT, MVD

def main():
    dataset = IDMT()
    feature_size, data = dataset.constructDataLoader('statistical')
    print(feature_size)
    print(len(data))
    print(data[0][0])
    print(data[0][1])
    print(data[1][1])

if __name__ == '__main__':
    main()