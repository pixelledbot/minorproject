import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# SAME MODEL STRUCTURE
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


classes = ['general','infectious','pharmaceutical','sharps']

model = WasteClassifier()
model.load_state_dict(torch.load("waste_classifier.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

image_path = r"C:\Users\pixel_t\Downloads\archive\data\test\infectious\vial_4.png"
image = Image.open(image_path).convert("RGB")

image = transform(image)
image = image.unsqueeze(0)

with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output,1)

print("Predicted class:", classes[predicted.item()])