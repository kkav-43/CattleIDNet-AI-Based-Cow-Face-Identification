import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# Define a simple CNN for cattle identification
class CattleCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(CattleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 56 * 56, 256)  # Adjust size based on input image
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# Instantiate the model
model = CattleCNN(num_classes=8)
print(model)
# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match CNN input
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for 3-channel images
])
dataset_path = "/content/drive/MyDrive/PROJECT C/cattle_dataset"
# Load dataset using ImageFolder
train_dataset = datasets.ImageFolder(root=f"{dataset_path}/train", transform=transform)
valid_dataset = datasets.ImageFolder(root=f"{dataset_path}/val", transform=transform)
test_dataset = datasets.ImageFolder(root=f"{dataset_path}/test", transform=transform)
# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
import torch.nn as nn
import torch.optim as optim
import torch
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Move model to GPU if available
model = CattleCNN(num_classes=8).to(device)
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Training function
def train(model, train_loader, valid_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.2f}%")
        # Validation phase
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        valid_accuracy = 100 * correct / total
        print(f"Validation Accuracy: {valid_accuracy:.2f}%")
# Start training
train(model, train_loader, valid_loader, epochs=10)
torch.save(model.state_dict(), "simple_cnn.pth")
print("Model saved successfully!")
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
# Function to evaluate the model
def evaluate_model(model, dataloader, device):
    model.eval()  # Set to evaluation mode
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred) * 100
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=1)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=1)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=1)
    conf_matrix = confusion_matrix(y_true, y_pred)
    # Print results in a clean format
    print("\n📊 Model Evaluation Results:")
    print(f"✅ Accuracy: {accuracy:.2f}%")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall: {recall:.4f}")
    print(f"✅ F1-score: {f1:.4f}")
    print("\n🟠 Confusion Matrix:")
    print(conf_matrix)
    return accuracy, precision, recall, f1, conf_matrix
# Load model and dataset (Ensure model is trained and loaded correctly)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Evaluate on test data
evaluate_model(model, test_loader, device)  # No extra print to avoid duplicate output
