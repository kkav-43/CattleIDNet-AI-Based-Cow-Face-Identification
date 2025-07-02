import os
import cv2
import numpy as np
# Map cattle ID (folder name) to label
label_map = {'E085':0, 'E263':1, 'E320':2, 'E333':3, 
             'E356':4, 'E416':5, 'E439':6, 'E446':7}
def load_images_from_folder(folder_path, img_size=(64, 64)):
    X = []
    y = []
    for cattle_id in os.listdir(folder_path):
        cattle_folder = os.path.join(folder_path, cattle_id)
        label = label_map[cattle_id]
        for img_file in os.listdir(cattle_folder):
            img_path = os.path.join(cattle_folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_size)
            X.append(img.flatten())  # Convert to 1D vector
            y.append(label)
    return np.array(X), np.array(y)
import os
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from utils import load_images_from_folder
# Create necessary directories
os.makedirs("features", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)
# Paths
paths = {
    "train":r"D:\PROJECT_C\CATTLE_DATASET\train",
    "valid": r"D:\PROJECT_C\CATTLE_DATASET\validation",
    "test": r"D:\PROJECT_C\CATTLE_DATASET\test"
}
# 1. Load and save features
for split in paths:
    X, y = load_images_from_folder(paths[split])
    np.save(f"features/{split}_features.npy", X)
    np.save(f"features/{split}_labels.npy", y)
    print(f"Saved {split} features: {X.shape}, Labels: {y.shape}")
# 2. Load features
X_train = np.load("features/train_features.npy")
y_train = np.load("features/train_labels.npy")
X_valid = np.load("features/valid_features.npy")
y_valid = np.load("features/valid_labels.npy")
X_test = np.load("features/test_features.npy")
y_test = np.load("features/test_labels.npy")
# 3. Train SVM
svm = SVC(kernel="linear", probability=True)
svm.fit(X_train, y_train)
# 4. Save model
joblib.dump(svm, "models/svm_model.pkl")
print("Model saved to models/svm_model.pkl")
# 5. Evaluate
y_pred = svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)
with open("results/accuracy.txt", "w") as f:
    f.write(f"Test Accuracy: {acc:.4f}\n")
print(f"Test Accuracy: {acc:.4f}")
# 6. Save confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap='Blues')
plt.savefig("results/confusion_matrix.png")
plt.close()
from sklearn.metrics import accuracy_score
# Train accuracy
y_train_pred = svm.predict(X_train)
train_acc = accuracy_score(y_train, y_train_pred)
print(f"Train Accuracy: {train_acc:.4f}")
# Validation accuracy
y_valid_pred = svm.predict(X_valid)
valid_acc = accuracy_score(y_valid, y_valid_pred)
print(f"Validation Accuracy: {valid_acc:.4f}")
