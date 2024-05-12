# from model import Net
from sklearn.model_selection import GridSearchCV
from models.nn import Shallow, Deep
from models.knn import knn_model
from models.svm import svm_model
from utils.data_utils import IDMT,MVD
from utils.training_utils import Trainer
import time

def main():
    # Grid search for KNN
    a = time.time()
    knn, X_train_scaled, y_train, y_test, y_pred, class_names, knn_for_gridsearch= knn_model()

    param_grid_knn = {
    'n_neighbors': [3, 5, 7, 10],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
    }

    # Set up the grid search for KNN
    grid_search_knn = GridSearchCV(estimator=knn_for_gridsearch, param_grid=param_grid_knn, cv=5, scoring='accuracy', verbose=1)
    grid_search_knn.fit(X_train_scaled, y_train)

    print("Best parameters for KNN:", grid_search_knn.best_params_)
    print("Best score for KNN:", grid_search_knn.best_score_)
    b = time.time()
    print(f'Time taken for kNN: {b-a:.2f}')
    # Best accuracy: 0.86 with {'metric': 'manhattan', 'n_neighbors': 10, 'weights': 'distance'}


    # Grid search for SVM
    a = time.time()
    svm, X_train_scaled, y_train, y_test, y_pred, class_names, svm_for_gridsearch = svm_model()

    param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'],  # Relevant for non-linear kernels
    'degree': [2, 3, 4]  # Relevant for 'poly' kernel
    }

    # Set up the grid search for SVM
    grid_search_svm = GridSearchCV(estimator=svm_for_gridsearch, param_grid=param_grid_svm, cv=5, scoring='accuracy', verbose=1)
    grid_search_svm.fit(X_train_scaled, y_train)

    # Best parameters and best score
    print("Best parameters for SVM:", grid_search_svm.best_params_)
    print("Best score for SVM:", grid_search_svm.best_score_)
    b = time.time()
    print(f'Time taken for SVM: {b-a:.2f}')
    # Best accuracy: 0.87 with {'C': 100, 'degree': 3, 'gamma': 'scale', 'kernel': 'poly'}
    # took 59 minutes

    # idmt = IDMT()
    
    # epochs = 10

    # features = ['ambient']
    # # models = [Deep]
    # models = [Shallow,Deep]
    # # param_2 = ['a','b']

    # for feature_set_type in features:

    #     # trainloader
    #     feature_size,trainloader = idmt.constructDataLoader(feature_set_type=feature_set_type)

    #     for model in models:

    #         # trainer = None
    #         # if model == 'shallow':
    #         #     trainer = Trainer(Shallow(input_dim=feature_size))

    #         print(f'[Grid Search] Feature: {feature_set_type} | Model: {model}')

    #         trainer = Trainer(model(input_dim=feature_size))

    #         trainer.training_epoch(epochs=epochs,trainloader=trainloader)

    #         # for p in param_2:
    #         #     print(f'[Grid Search] Feature: {feature_set_type} | Model: {model} | Param: {p}')

    #             # train model

    #             # store model params + loss

    # pass

if __name__ == '__main__':

    main()