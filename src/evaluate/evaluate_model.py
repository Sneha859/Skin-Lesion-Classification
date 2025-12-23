# src/evaluate/evaluate_model.py
"""
Evaluate saved model on test set and produce Confusion Matrix + classification report + ROC curves.
Usage:
python src/evaluate/evaluate_model.py
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import itertools

from src.utils.config import CHECKPOINTS_DIR, PLOTS_DIR
from src.utils.dataset_loader import get_train_val_test_datasets

def plot_confusion_matrix(cm, classes, outpath):
    plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def main():
    train_ds, val_ds, test_ds, class_names = get_train_val_test_datasets()
    model_path = Path(CHECKPOINTS_DIR) / "efficientnet_b4_best.h5"
    if not model_path.exists():
        model_path = Path(CHECKPOINTS_DIR) / "efficientnet_b4_final.h5"
    model = tf.keras.models.load_model(model_path)
    print("Loaded model:", model_path)

    # Collect y_true, y_pred probabilities
    y_true = []
    y_prob = []
    for x_batch, y_batch in test_ds:
        y_true.extend(np.argmax(y_batch.numpy(), axis=1).tolist())
        probs = model.predict(x_batch)
        y_prob.extend(probs.tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.argmax(y_prob, axis=1)

    # classification report
    print(classification_report(y_true, y_pred, target_names=class_names))

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, Path(PLOTS_DIR)/"confusion_matrix.png")
    print("Confusion matrix saved.")

    # ROC curves (one-vs-rest)
    from sklearn.preprocessing import label_binarize
    y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
    plt.figure(figsize=(10,8))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.savefig(Path(PLOTS_DIR)/"roc_curves.png")
    plt.close()
    print("ROC curves saved.")

if __name__ == "__main__":
    main()
