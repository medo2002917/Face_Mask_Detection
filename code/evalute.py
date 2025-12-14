import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

MODEL_PATH = "../saved_model/best_model.h5"
RESULTS_DIR = "../results"
DATA_DIR = "../data"

IMG_SIZE = 128
CLASS_NAMES = ["No Mask", "Mask"]

os.makedirs(RESULTS_DIR, exist_ok=True)

model = keras.models.load_model(MODEL_PATH)

X_test = []
y_test = []

for label, class_name in enumerate(["without_mask", "with_mask"]):
    class_path = os.path.join(DATA_DIR, class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        X_test.append(img)
        y_test.append(label)

X_test = np.array(X_test)
y_test = np.array(y_test)

y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.close()

num_samples = 6
indices = np.random.choice(len(X_test), num_samples, replace=False)

plt.figure(figsize=(12, 6))
for i, idx in enumerate(indices):
    plt.subplot(2, 3, i + 1)
    plt.imshow(X_test[idx])
    true_label = CLASS_NAMES[y_test[idx]]
    pred_label = CLASS_NAMES[y_pred[idx]]
    plt.title(f"T: {true_label}\nP: {pred_label}")
    plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "sample_predictions.png"))
plt.close()

print("✅ Evaluation complete. Results saved in /results")