from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.sql_utils import DB
import pandas as pd

def ensemble_models():
    db = DB()
    idmt_df = db.downloadDF('IDMT_statistical_features')
    mvd_df = db.downloadDF('MVD_statistical_features')
    df = pd.concat([idmt_df, mvd_df], ignore_index=True)
    idmt_X = idmt_df[['mode_var', 's', 'i', 'variance', 'gstd_var', 'ent', 'fc', 'azcr']]  # Features for IDMT
    mvd_X = mvd_df[['mode_var', 's', 'i', 'variance', 'gstd_var', 'ent', 'fc', 'azcr']]    # Features for MVD
    combined_X = df[['mode_var', 's', 'i', 'variance', 'gstd_var', 'ent', 'fc', 'azcr']]   # Features for combined dataset

    idmt_df = db.downloadDF('IDMT_features')
    mvd_df = db.downloadDF('MVD_features')
    df = pd.concat([idmt_df, mvd_df], ignore_index=True)
    idmt_y = idmt_df['class']          # Target variable for IDMT
    mvd_y = mvd_df['class']            # Target variable for MVD
    combined_y = df['class']           # Target variable for combined dataset
    class_names = combined_y.unique()

    X_train, X_test, y_train, y_test = train_test_split(idmt_X, idmt_y, test_size=0.3, random_state=42)

    rf1 = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=10)
    rf1.fit(X_train, y_train)
    y_pred = rf1.predict(X_test)
    acc1 = accuracy_score(y_test, y_pred)

    X_train, X_test, y_train, y_test = train_test_split(mvd_X, mvd_y, test_size=0.3, random_state=42)

    rf2 = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=10)
    rf2.fit(X_train, y_train)
    y_pred = rf2.predict(X_test)
    acc2 = accuracy_score(y_test, y_pred)

    acc = (acc1 + acc2) / 2
    #prob = knn.predict_proba(X_test)
    

    return acc#, prob

def main():
    acc = ensemble_models()
    print(f'Accuracy: {acc:.2f}')
    #print(prob)
          
if __name__ == '__main__':
    main()

