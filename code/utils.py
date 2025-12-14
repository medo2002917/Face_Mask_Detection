import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools

def plot_and_save_history(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Accuracy curve
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{out_dir}/accuracy_curve.png")
    plt.close()

    # Loss curve
    plt.figure()
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{out_dir}/loss_curve.png")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, classes, out_path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > thresh else "black"
        plt.text(j, i, cm[i, j], ha="center", color=color)

    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_sample_predictions(model, X_test, y_test, out_path, n=12):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    idxs = np.random.choice(len(X_test), n, replace=False)

    plt.figure(figsize=(12, 12))
    for i, idx in enumerate(idxs):
        img = X_test[idx]
        true = y_test[idx]
        pred = np.argmax(model.predict(img.reshape(1, 128, 128, 3)))

        plt.subplot(4, 4, i + 1)
        plt.imshow(img)
        plt.title(f"T:{true}  P:{pred}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
