from models.knn import knn_model
from sklearn.metrics import classification_report, accuracy_score

def main():
    knn, X_train_scaled, y_train, y_test, y_pred = knn_model()
    print("Accuracy:", accuracy_score(y_test, y_pred))
    #print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
if __name__ == '__main__':
    main()