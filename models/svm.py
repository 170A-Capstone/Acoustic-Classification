from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from utils.sql_utils import DB

def svm_model():
    db = DB()
    df = db.downloadDF('IDMT_statistical_features')
    X = df[['mode_var', 'k', 's', 'mean', 'i', 'g', 'h', 'dev', 'var', 'variance', 'std', 'gstd_var', 'ent']]  # Features

    df = db.downloadDF('IDMT_features')
    y = df['class']          # Target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the SVM model
    svm_model = SVC(kernel='rbf')  # 'rbf' is good for non-linear problems, change as needed
    svm_model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = svm_model.predict(X_test_scaled)

    return svm_model, X_train_scaled, y_train, y_test, y_pred