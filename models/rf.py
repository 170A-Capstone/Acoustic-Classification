from utils.sql_utils import DB
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

def rf_model():
    db = DB()
    idmt_df = db.downloadDF('IDMT_statistical_features')
    mvd_df = db.downloadDF('MVD_statistical_features')
    df = pd.concat([idmt_df, mvd_df], ignore_index=True)
    X = df[['mode_var', 's', 'i', 'variance', 'gstd_var', 'ent', 'fc', 'azcr']]  # Features

    idmt_df = db.downloadDF('IDMT_features')
    mvd_df = db.downloadDF('MVD_features')
    df = pd.concat([idmt_df, mvd_df], ignore_index=True)
    y = df['class']          # Target variable
    class_names = y.unique()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=10)
    rf.fit(X_train_scaled, y_train)

    y_pred = rf.predict(X_test_scaled)

    rf_for_gridsearch = RandomForestClassifier()

    return rf, X_train_scaled, y_train, y_test, y_pred, class_names, rf_for_gridsearch

