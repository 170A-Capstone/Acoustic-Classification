# SVM model testing
from models.svm import svm_model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrix(cm, title='Confusion Matrix', class_names=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(title)
    plt.show()

def main():
    # Get the model and predictions
    svm, X_train_scaled, y_train,y_test, y_pred, class_names, svm_for_gridsearch = svm_model()
    
    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    #print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm_svm = confusion_matrix(y_test, y_pred, labels=class_names)
    cm_svm = cm_svm.astype('float') / cm_svm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_svm, title='Confusion Matrix for SVM', class_names=class_names)
    

if __name__ == "__main__":
    main()