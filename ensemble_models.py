from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.sql_utils import DB
import pandas as pd

def ensemble_models():
    db = DB()
    idmt_df = db.downloadDF('IDMT_statistical_features')
    mvd_df = db.downloadDF('MVD_statistical_features')
    df = pd.concat([idmt_df, mvd_df], ignore_index=True)
    idmt_X = idmt_df[['mode_var', 'k', 's', 'mean', 'i', 'g', 'h', 'dev', 'var', 'variance', 'std', 'gstd_var', 'ent']]  # Features for IDMT
    mvd_X = mvd_df[['mode_var', 'k', 's', 'mean', 'i', 'g', 'h', 'dev', 'var', 'variance', 'std', 'gstd_var', 'ent']]    # Features for MVD
    combined_X = df[['mode_var', 'k', 's', 'mean', 'i', 'g', 'h', 'dev', 'var', 'variance', 'std', 'gstd_var', 'ent']]   # Features for combined dataset

    idmt_df = db.downloadDF('IDMT_features')
    mvd_df = db.downloadDF('MVD_features')
    df = pd.concat([idmt_df, mvd_df], ignore_index=True)
    idmt_y = idmt_df['class']          # Target variable for IDMT
    mvd_y = mvd_df['class']            # Target variable for MVD
    combined_y = df['class']           # Target variable for combined dataset
    class_names = combined_y.unique()

    X_train, X_test, y_train, y_test = train_test_split(idmt_X, idmt_y, test_size=0.3, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=10, metric='manhattan', weights='distance')
    knn.fit(X_train, y_train)

    X_train, X_test, y_train, y_test = train_test_split(mvd_X, mvd_y, test_size=0.3, random_state=42)

    svc = SVC(kernel='poly', C=100, degree=3, gamma='scale', probability=True)
    svc.fit(X_train, y_train)

    X_train, X_test, y_train, y_test = train_test_split(combined_X, combined_y, test_size=0.3, random_state=42)
    ensemble = VotingClassifier(estimators=[('knn', knn), ('svc', svc)], voting='soft')
    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    #prob = knn.predict_proba(X_test)
    

    return acc#, prob

def main():
    acc = ensemble_models()
    print(f'Accuracy: {acc:.2f}')
    #print(prob)
          
if __name__ == '__main__':
    main()

