import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import timm
from einops import rearrange
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
data_dir = '/content/drive/MyDrive/PROJECT C/cattle_dataset'
train_data = ImageFolder(f"{data_dir}/train", transform=transform)
valid_data = ImageFolder(f"{data_dir}/val", transform=transform)
test_data  = ImageFolder(f"{data_dir}/test", transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader  = DataLoader(test_data, batch_size=32)
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )
    def forward(self, x):
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x
class CattleIDNet(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        # ✅ Use EfficientNet-B0 pretrained
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True, features_only=True)
        self.backbone_out = self.backbone.feature_info[-1]['num_chs']  # Usually 1280

        self.project = nn.Conv2d(self.backbone_out, 256, kernel_size=1)
        self.transformer = TransformerBlock(dim=256, heads=4, mlp_dim=512)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)[-1]               # (B, C, H, W)
        x = self.project(features)                    # (B, 256, H, W)
        x = rearrange(x, 'b c h w -> b (h w) c')      # (B, N, 256)
        x = self.transformer(x)                       # Transformer
        x = x.mean(dim=1)                             # Global avg pool
        return self.classifier(x)
model = CattleIDNet(num_classes=8).to("cuda")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def train_epoch(loader, model, optimizer):
    model.train()
    total_loss, correct = 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.cuda(), labels.cuda()
        preds = model(imgs)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (preds.argmax(1) == labels).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)
def evaluate(loader, model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.cuda(), labels.cuda()
            preds = model(imgs)
            correct += (preds.argmax(1) == labels).sum().item()
    return correct / len(loader.dataset)
import torch

# Function to save the model and optimizer
def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path="model_checkpoint.pth"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")

# Train loop with saving the model every 5 epochs
train_losses, val_losses = [], []  # Tracking losses
train_accuracies, val_accuracies = [], []  # Tracking accuracies

for epoch in range(10):  # Training for 10 epochs (you can change this as needed)
    # Train the model for one epoch
    train_loss, train_acc = train_epoch(train_loader, model, optimizer)

    # Evaluate the model on the validation set
    val_loss, val_acc = evaluate_loss_and_acc(valid_loader, model)

    # Step the scheduler based on validation loss after each epoch
    scheduler.step(val_loss)

    # Log metrics for visualization
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    # Print the stats for this epoch
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Save the model every 5 epochs
    if (epoch + 1) % 5 == 0:
        save_checkpoint(model, optimizer, epoch + 1, val_loss, checkpoint_path=f"model_epoch_{epoch+1}.pth")

# Optionally, you can save the final model at the end of all epochs
save_checkpoint(model, optimizer, 10, val_loss, checkpoint_path="final_model.pth")
# ✅ Plot Loss and Accuracy
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss Curve For Hybrid Model ")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Acc")
plt.plot(val_accuracies, label="Val Acc")
plt.title("Accuracy Curve For Hybrid Model")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ✅ Evaluate on Known Test Set
_, test_known_acc = evaluate_loss_and_acc(test_known_loader, model)
print(f"\n✅ Final Known Test Accuracy: {test_known_acc:.4f}")
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

confidence_threshold = 0.99
known_classes = ['E085', 'E263', 'E320', 'E333', 'E356', 'E416', 'E439', 'E446']
unknown_class_index = len(known_classes)  # e.g., index 8

all_preds = []
all_labels = []

# Evaluate only on known test data
with torch.no_grad():
    for imgs, labels in test_known_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        probabilities = F.softmax(outputs, dim=1)

        max_probs, preds = torch.max(probabilities, 1)

        # Assign to 'Unknown' if below threshold
        preds[max_probs < confidence_threshold] = unknown_class_index

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Filter out predictions labeled as "Unknown"
valid_indices = all_preds != unknown_class_index
filtered_preds = all_preds[valid_indices]
filtered_labels = all_labels[valid_indices]

# Confusion Matrix (only known class predictions)
cm = confusion_matrix(filtered_labels, filtered_preds, labels=list(range(len(known_classes))))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=known_classes)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix (Known Class Predictions Only)")
plt.show()

# Classification Report
report = classification_report(filtered_labels, filtered_preds, target_names=known_classes, digits=4)
print("Classification Report (excluding Unknown predictions):\n")
print(report)

# Optional: Save to file
with open("classification_report_known_only_filtered.txt", "w") as f:
    f.write(report)
