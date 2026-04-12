import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
import os
import copy


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔧 Using device:", device)

    DATASET_PATH = "dataset_split"
    IMG_SIZE = 224

    # ---------------- STRONG AUGMENTATION ----------------
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATASET_PATH, "train"), transform=train_transform)
    val_dataset   = datasets.ImageFolder(os.path.join(DATASET_PATH, "val"), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    # ---------------- MODEL ----------------
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # 🔥 STRONG FINE-TUNING (BEST PART)
    for name, param in model.named_parameters():
        if "layer3" in name or "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, 4)
    model = model.to(device)

    # ---------------- LOSS (LABEL SMOOTHING = BETTER ACCURACY) ----------------
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ---------------- OPTIMIZER (BETTER THAN ADAM) ----------------
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0003,
        weight_decay=1e-4
    )

    # ---------------- SCHEDULER (MODERN BEST PRACTICE) ----------------
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)

    epochs = 8
    best_acc = 0
    best_model = copy.deepcopy(model.state_dict())

    SAVE_PATH = "waste_classifier_best.pth"

    for epoch in range(epochs):

        print(f"\n🚀 Epoch {epoch+1}/{epochs}")

        model.train()
        correct, total = 0, 0
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

        scheduler.step()

        print(f"Loss: {running_loss:.4f}")
        print(f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        # ---------------- SAVE BEST ----------------
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, SAVE_PATH)
            print("💾 Best model saved!")

    model.load_state_dict(best_model)
    torch.save(model.state_dict(), SAVE_PATH)

    print("\n🎯 Training Complete")
    print("🏆 BEST MODEL SAVED:", SAVE_PATH)


if __name__ == "__main__":
    main()