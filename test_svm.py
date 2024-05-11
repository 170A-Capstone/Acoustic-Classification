# SVM model testing
from models.svm import svm_model
from sklearn.metrics import classification_report, accuracy_score

def main():
    # Get the model and predictions
    svm, X_train_scaled, y_train,y_test, y_pred = svm_model()
    
    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    #print("\nClassification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()