from utils.sql_utils import DB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

def knn_model():
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

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=10, metric='manhattan', weights='distance')
    knn.fit(X_train_scaled, y_train)

    y_pred = knn.predict(X_test_scaled)

    knn_for_gridsearch = KNeighborsClassifier()

    return knn, X_train_scaled, y_train, y_test, y_pred, class_names, knn_for_gridsearch

