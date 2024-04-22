from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def k_nearest_neighbors(features, traget, num):
    x_train, x_test, y_train, y_test = train_test_split(features, traget, test_size = 0.2)

    knn = KNeighborsClassifier(n_neighbors = num)

    knn.fit(x_train, y_train)

    predictions = knn.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)

    print("knn accuracy: ", accuracy)