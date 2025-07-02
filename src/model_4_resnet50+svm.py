import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image  # Import PIL for image conversion
# Load Pre-trained ResNet-50 Model
device = "cuda" if torch.cuda.is_available() else "cpu"
resnet = models.resnet50(pretrained=True).to(device)
resnet.eval()  # Set model to evaluation mode
# Remove last layer (for feature extraction)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
# Transformation for input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Paths
cropped_faces_root = r"D:\PROJECT_C\CATTLE_DATASET"
output_feature_path = r"H:\PROJECT C\using resnet 50\features"
# Ensure output directory exists
os.makedirs(output_feature_path, exist_ok=True)
def extract_features():
    """Extracts features from cattle face images using ResNet-50."""
    features = []
    labels = []
    label_map = {}
    class_index = 0
    for split in ["train", "test", "validation"]:
        split_path = os.path.join(cropped_faces_root, split)
        # Skip missing dataset folders
        if not os.path.exists(split_path):
            print(f"⚠️ Skipping missing folder: {split_path}")
            continue  
        for cattle_name in os.listdir(split_path):
            cattle_path = os.path.join(split_path, cattle_name)
            if not os.path.isdir(cattle_path):
                continue  # Skip non-directory files
            if cattle_name not in label_map:
                label_map[cattle_name] = class_index
                class_index += 1
            for img_name in os.listdir(cattle_path):
                img_path = os.path.join(cattle_path, img_name)
                # Read and preprocess image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"⚠️ Skipping corrupt image: {img_path}")
                    continue
                try:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert OpenCV BGR to RGB
                    img = Image.fromarray(img)  # Convert NumPy array to PIL image
                    img = transform(img).unsqueeze(0).to(device)  # Apply transforms
                except Exception as e:
                    print(f"⚠️ Error processing {img_path}: {e}")
                    continue
                # Extract features
                with torch.no_grad():
                    feature = resnet(img).cpu().numpy().flatten()
                features.append(feature)
                labels.append(label_map[cattle_name])
    # Save extracted features and labels
    np.save(os.path.join(output_feature_path, "features.npy"), np.array(features))
    np.save(os.path.join(output_feature_path, "labels.npy"), np.array(labels))
    np.save(os.path.join(output_feature_path, "label_mapping.npy"), label_map)
    print("✅ Features extracted and saved!")
# Run feature extraction
extract_features()
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # For saving the model
# Load extracted features
features = np.load(r"H:\PROJECT C\using resnet 50\features\features.npy")
labels = np.load(r"H:\PROJECT C\using resnet 50\features\labels.npy")
# Split data into train & test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
# Train SVM classifier
svm_classifier = SVC(kernel='linear', probability=True)
svm_classifier.fit(X_train, y_train)
# Evaluate model
y_pred = svm_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ SVM Accuracy: {accuracy * 100:.2f}%")
# Ensure the model directory exists
model_dir = r"H:\PROJECT C\using resnet 50\model"
os.makedirs(model_dir, exist_ok=True)  # ✅ This ensures the directory exists
# Save trained model
model_path = os.path.join(model_dir, "cattle_svm_model.pkl")
joblib.dump(svm_classifier, model_path)
print(f"✅ Model saved successfully at {model_path}!")
from PIL import Image  # Import PIL
import cv2
import torch
import numpy as np                             
import joblib
import torchvision.transforms as transforms
from torchvision import models
# Load trained SVM model
model_path = r"H:\PROJECT C\using resnet 50\model\cattle_svm_model.pkl"
svm_classifier = joblib.load(model_path)
# Load ResNet-50 for feature extraction
device = "cuda" if torch.cuda.is_available() else "cpu"
resnet = models.resnet50(pretrained=True).to(device)
resnet.eval()
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove final layer
# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# Load label mapping
label_map = np.load(r"H:\PROJECT C\using resnet 50\features\label_mapping.npy", allow_pickle=True).item()
reverse_label_map = {v: k for k, v in label_map.items()}  # Reverse lookup for cattle names
def predict_cattle(image_path):
    """Predicts cattle identity from an image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Unable to read {image_path}")
        return
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img_pil = Image.fromarray(img)  # Convert NumPy array to PIL Image
    img_tensor = transform(img_pil).unsqueeze(0).to(device)  # Apply transforms
    with torch.no_grad():
        feature = resnet(img_tensor).cpu().numpy().flatten().reshape(1, -1)
    # Predict cattle identity
    predicted_label = svm_classifier.predict(feature)[0]
    cattle_name = reverse_label_map[predicted_label]
    print(f"✅ Predicted Cattle: {cattle_name}")
# Test with a new image
test_image = r"H:\PROJECT C\FINAL\FINAL\dataset_balanced_split\test\E333\IMG_20250110_161344_Burst14.jpg"
predict_cattle(test_image)
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # For saving the model
# Load extracted features
features = np.load(r"H:\PROJECT C\using resnet 50\features\features.npy")
labels = np.load(r"H:\PROJECT C\using resnet 50\features\labels.npy")
# Split data into train (70%), validation (15%), and test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
# Train SVM classifier (Try both linear and RBF kernels)
svm_classifier = SVC(kernel='linear', probability=True)  # Change 'linear' to 'rbf' for RBF kernel
svm_classifier.fit(X_train, y_train)
# Evaluate model on Train, Validation, and Test sets
train_acc = accuracy_score(y_train, svm_classifier.predict(X_train))
valid_acc = accuracy_score(y_valid, svm_classifier.predict(X_valid))
test_acc = accuracy_score(y_test, svm_classifier.predict(X_test))
print(f"✅ Train Accuracy: {train_acc * 100:.2f}%")
print(f"✅ Validation Accuracy: {valid_acc * 100:.2f}%")
print(f"✅ Test Accuracy: {test_acc * 100:.2f}%")
# Ensure the model directory exists
model_dir =  r"H:\PROJECT C\using resnet 50\model"
os.makedirs(model_dir, exist_ok=True)
# Save trained model
model_path = os.path.join(model_dir, "cattle_svm_model2.pkl")
joblib.dump(svm_classifier, model_path)
print(f"✅ Model saved successfully at {model_path}!")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# Predict labels for the test set
y_pred = svm_classifier.predict(X_test)
# Compute confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=set(y_test), yticklabels=set(y_test))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix Heatmap")
plt.show()
# Print classification report
print("🔹 Classification Report:\n")
print(classification_report(y_test, y_pred))
