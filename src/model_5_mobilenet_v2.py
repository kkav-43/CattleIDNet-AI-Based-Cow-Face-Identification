import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Parameters
num_classes = 8  # Change according to your dataset
batch_size = 32
learning_rate = 0.0001
epochs = 10
image_size = 224
# Data transformations
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
data_dir = '/content/drive/MyDrive/PROJECT C/cattle_dataset'
train_data = ImageFolder(f"{data_dir}/train", transform=transform)
valid_data = ImageFolder(f"{data_dir}/val", transform=transform)
test_data  = ImageFolder(f"{data_dir}/test", transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader  = DataLoader(test_data, batch_size=32)
def create_mobilenet_v2(num_classes):
    # Load pre-trained MobileNetV2
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    # Modify classifier for our number of classes
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    return model
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    model.to(device)
    best_acc = 0.0
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / total
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "mobilenet_v2_best.pth")    
    return model
if __name__ == "__main__":
    # Create MobileNetV2 model
    model = create_mobilenet_v2(num_classes)
    print("MobileNetV2 Architecture:")
    print(model)    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)   
    # Train the model
    trained_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        epochs
    )
6.EfficientNet B0
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
num_classes = 10  # Change according to your dataset
batch_size = 32
learning_rate = 0.001
epochs = 10
image_size = 224

# Data transformations
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
data_dir = '/content/drive/MyDrive/PROJECT C/cattle_dataset'
train_data = ImageFolder(f"{data_dir}/train", transform=transform)
valid_data = ImageFolder(f"{data_dir}/val", transform=transform)
test_data  = ImageFolder(f"{data_dir}/test", transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader  = DataLoader(test_data, batch_size=32)

def create_efficientnet_b0(num_classes):
    # Load pre-trained EfficientNetB0
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    
    # Modify classifier for our number of classes
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    model.to(device)
    best_acc = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / total
        
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "efficientnet_b0_best.pth")
    
    return model

def save_model_as_pth(model, filename="efficientnet_b0.pth"):
    torch.save(model.state_dict(), filename)
    print(f"Model saved as {filename}")

def load_model_from_pth(filename, num_classes):
    model = create_efficientnet_b0(num_classes)
    model.load_state_dict(torch.load(filename))
    model.eval()
    return model
if __name__ == "__main__":
    # Create EfficientNetB0 model
    model = create_efficientnet_b0(num_classes)
    print("EfficientNetB0 Architecture:")
    print(model)    
    # Save the model with .pth extension
    save_model_as_pth(model)    
    # Example of training setup (uncomment when ready to train)    
    # Initialize criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Train the model
    trained_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        epochs
    )
