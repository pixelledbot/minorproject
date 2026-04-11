import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# EXACT SAME MODEL USED IN TRAINING
class WasteClassifier(nn.Module):
    def __init__(self):
        super(WasteClassifier, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*28*28,128),
            nn.ReLU(),
            nn.Linear(128,4)
        )

    def forward(self,x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

test_dataset = datasets.ImageFolder(
    root="data/test",
    transform=transform
)

test_loader = DataLoader(test_dataset,batch_size=32,shuffle=False)

model = WasteClassifier()
model.load_state_dict(torch.load("waste_classifier.pth"))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs,1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print("Test Accuracy:", accuracy,"%")