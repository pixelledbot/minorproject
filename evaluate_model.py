import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

IMG_SIZE = 224  # MUST MATCH TRAINING

# ---------------- TRANSFORM (MUST MATCH TRAINING) ----------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ---------------- DATASET ----------------
test_dataset = datasets.ImageFolder(
    root="dataset_split/test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

# ---------------- MODEL ----------------
model = resnet18(weights=None)  # IMPORTANT: no re-download
model.fc = nn.Linear(model.fc.in_features, 4)

# Load trained weights
model.load_state_dict(torch.load("waste_classifier_best.pth", map_location=device))

model = model.to(device)
model.eval()

# ---------------- TESTING ----------------
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print("\n🎯 Test Accuracy:", round(accuracy, 2), "%")
print("✅ Classes:", test_dataset.classes)