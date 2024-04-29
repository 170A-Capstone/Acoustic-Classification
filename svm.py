from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def svm_model(features, target):
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size = 0.2)

    model = svm.SVC(kernel = 'linear')

    # other options for kernel:
    # Polynomial Kernel eg:(kernel = 'poly', degree = 3)
    # Radial Basis Function Kernel (kernel = 'rbf', gamma = 'scale')
    # Sigmoid Kernel (kernel = 'sigmoid')

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy: ", accuracy)