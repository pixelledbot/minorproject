import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
import os
import copy
import time


def main():

    # ---------------- DEVICE ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔧 Using device:", device)

    DATASET_PATH = "dataset_split"

    # ---------------- TRANSFORMS ----------------
    train_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])

    # ---------------- DATASETS ----------------
    train_dataset = datasets.ImageFolder(os.path.join(DATASET_PATH, "train"), transform=train_transform)
    val_dataset   = datasets.ImageFolder(os.path.join(DATASET_PATH, "val"), transform=val_test_transform)
    test_dataset  = datasets.ImageFolder(os.path.join(DATASET_PATH, "test"), transform=val_test_transform)

    # ---------------- DATALOADERS ----------------
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # ---------------- MODEL ----------------
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 4)
    model = model.to(device)

    # ---------------- LOSS ----------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # ---------------- TRAINING SETTINGS ----------------
    epochs = 5   # 🔥 REDUCED TO 5
    best_acc = 0
    best_model = copy.deepcopy(model.state_dict())

    SAVE_PATH = "waste_classifier_final.pth"

    # ---------------- TRAIN LOOP ----------------
    for epoch in range(epochs):

        print(f"\n🚀 Epoch {epoch+1}/{epochs}")

        model.train()
        total, correct = 0, 0
        running_loss = 0

        for i, (images, labels) in enumerate(train_loader):

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            if i % 20 == 0:
                print(f"   Batch {i}/{len(train_loader)}")

        train_acc = 100 * correct / total

        # ---------------- VALIDATION ----------------
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_acc = 100 * val_correct / val_total

        print(f"Loss: {running_loss:.4f}")
        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # ---------------- SAVE BEST MODEL ----------------
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, SAVE_PATH)
            print("💾 Best model saved!")

    # ---------------- FINAL SAVE (VERY IMPORTANT) ----------------
    model.load_state_dict(best_model)
    torch.save(model.state_dict(), SAVE_PATH)

    print("\n🎯 Training Complete")
    print("💾 Final model saved at:", SAVE_PATH)


# ---------------- WINDOWS SAFE ENTRY ----------------
if __name__ == "__main__":
    main()