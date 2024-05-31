from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from models.rf import rf_model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def main():
    param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
    }

    rf, X_train_scaled, y_train, y_test, y_pred, class_names, rf_for_gridsearch = rf_model()

    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), 
                           param_grid=param_grid, 
                           cv=3, 
                           verbose=2, 
                           n_jobs=-1)
    
    grid_search.fit(X_train_scaled, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.2f}")

    
if __name__ == '__main__':
    main()