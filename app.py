from flask import Flask, render_template, request, send_from_directory, redirect
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os, sqlite3, datetime

app = Flask(__name__)

# ---------------- DB ----------------
conn = sqlite3.connect("database.db")
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS history (
id INTEGER PRIMARY KEY AUTOINCREMENT,
filename TEXT,
prediction TEXT,
confidence REAL,
timestamp TEXT
)
""")
conn.commit()
conn.close()

# ---------------- MODEL ----------------
class WasteClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*28*28,128),
            nn.ReLU(),
            nn.Linear(128,4)
        )
    def forward(self,x):
        return self.fc_layers(self.conv_layers(x))

model = WasteClassifier()
model.load_state_dict(torch.load("waste_classifier.pth", map_location="cpu"))
model.eval()

classes = ["general","infectious","pharmaceutical","sharps"]

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")

    if file is None or file.filename == "":
        return "No file uploaded"

    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    img = Image.open(filepath).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    prediction = classes[pred.item()]
    confidence = round(conf.item()*100,2)

    # All class confidence
    all_probs = {
        classes[i]: round(probs[0][i].item()*100,2)
        for i in range(4)
    }

    # Risk
    if prediction in ["infectious","sharps"]:
        risk = "High"
    elif prediction == "pharmaceutical":
        risk = "Medium"
    else:
        risk = "Low"

    # Instructions
    instructions = {
        "general": "Dispose in green bin.",
        "infectious": "Dispose in yellow biohazard bag.",
        "pharmaceutical": "Dispose in blue chemical bin.",
        "sharps": "Dispose in puncture-proof container."
    }

    # Save
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("INSERT INTO history (filename,prediction,confidence,timestamp) VALUES (?,?,?,?)",
              (file.filename,prediction,confidence,str(datetime.datetime.now())))
    conn.commit()
    conn.close()

    return render_template("result.html",
                           prediction=prediction,
                           confidence=confidence,
                           filename=file.filename,
                           risk=risk,
                           instruction=instructions[prediction],
                           all_probs=all_probs)

@app.route("/dashboard")
def dashboard():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM history")
    total = c.fetchone()[0]

    c.execute("SELECT prediction, COUNT(*) FROM history GROUP BY prediction")
    data = c.fetchall()

    c.execute("SELECT * FROM history ORDER BY id DESC")
    logs = c.fetchall()

    conn.close()

    return render_template("dashboard.html",
                           total=total,
                           data=data,
                           logs=logs)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

app.run(debug=True)