from models.knn import knn_model
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
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
    knn, X_train_scaled, y_train, y_test, y_pred, class_names, knn_for_gridsearch = knn_model()
    print("Accuracy:", accuracy_score(y_test, y_pred))
    #print("\nClassification Report:\n", classification_report(y_test, y_pred))


    cm_knn = confusion_matrix(y_test, y_pred, labels=class_names)
    #cm_knn = confusion_matrix(y_test, y_pred)
    # 打印混淆矩阵
    #print(cm)
    cm_knn = cm_knn.astype('float') / cm_knn.sum(axis=1)[:, np.newaxis]
    #print(cm_normalized)
    plot_confusion_matrix(cm_knn, title='Confusion Matrix for KNN', class_names=class_names)

    
    
if __name__ == '__main__':
    main()