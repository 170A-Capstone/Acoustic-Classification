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
    X = df[['mode_var', 'k', 's', 'mean', 'i', 'g', 'h', 'dev', 'var', 'variance', 'std', 'gstd_var', 'ent']]  # Features

    idmt_df = db.downloadDF('IDMT_features')
    mvd_df = db.downloadDF('MVD_features')
    df = pd.concat([idmt_df, mvd_df], ignore_index=True)
    y = df['class']          # Target variable
    class_names = y.unique()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=10, metric='manhattan', weights='distance')
    svc = SVC(kernel='poly', C=100, degree=3, gamma='scale', probability=True)

    ensemble = VotingClassifier(estimators=[('knn', knn), ('svc', svc)], voting='soft')
    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    return acc

def main():
    acc = ensemble_models()
    print(f'Accuracy: {acc:.2f}')
          
if __name__ == '__main__':
    main()

