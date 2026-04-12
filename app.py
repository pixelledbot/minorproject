from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os, sqlite3, datetime, time
import cv2
import numpy as np

app = Flask(__name__)

# ---------------- DB ----------------
def init_db():
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

init_db()

# ---------------- MODEL (🔥 RESNET FIX) ----------------
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 4)

model.load_state_dict(torch.load("waste_classifier_best.pth", map_location="cpu"))
model.eval()

classes = ["general","infectious","pharmaceutical","sharps"]

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# ---------------- 🔥 GRAD-CAM (FIXED FOR RESNET) ----------------
def generate_heatmap(model, image_tensor, class_idx):
    gradients = []
    activations = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # 🔥 correct layer for ResNet
    target_layer = model.layer4[-1]

    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    model.zero_grad()

    loss = output[0, class_idx]
    loss.backward()

    grads = gradients[0].detach().numpy()[0]
    acts = activations[0].detach().numpy()[0]

    weights = np.mean(grads, axis=(1, 2))

    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)

    if cam.max() != 0:
        cam = cam / cam.max()

    cam = cv2.resize(cam, (128, 128))

    handle_f.remove()
    handle_b.remove()

    return cam

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

    # SAVE IMAGE
    os.makedirs("static/uploads", exist_ok=True)
    filename = str(int(time.time())) + "_" + file.filename
    filepath = os.path.join("static/uploads", filename)
    file.save(filepath)

    # PREPROCESS
    img = Image.open(filepath).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    # PREDICT
    outputs = model(img_tensor)
    probs = torch.softmax(outputs, dim=1)
    conf, pred = torch.max(probs, 1)

    prediction = classes[pred.item()]
    confidence = round(conf.item()*100,2)

    # ALL PROBS
    all_probs = {
        classes[i]: round(probs[0][i].item()*100,2)
        for i in range(4)
    }

    # RISK
    if prediction in ["infectious","sharps"]:
        risk = "High"
    elif prediction == "pharmaceutical":
        risk = "Medium"
    else:
        risk = "Low"

    # INSTRUCTIONS
    instructions = {
        "general": "Dispose in green bin.",
        "infectious": "Dispose in yellow biohazard bag.",
        "pharmaceutical": "Dispose in blue chemical bin.",
        "sharps": "Dispose in puncture-proof container."
    }

    # ---------------- 🔥 HEATMAP ----------------
    heatmap = generate_heatmap(model, img_tensor, pred.item())

    original = cv2.imread(filepath)
    original = cv2.resize(original, (128, 128))

    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

    heatmap_filename = "heatmap_" + filename
    heatmap_path = os.path.join("static/uploads", heatmap_filename)
    cv2.imwrite(heatmap_path, overlay)

    # SAVE TO DB
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO history (filename,prediction,confidence,timestamp) VALUES (?,?,?,?)",
        (filename,prediction,confidence,str(datetime.datetime.now()))
    )
    conn.commit()
    conn.close()

    return render_template("result.html",
                           prediction=prediction,
                           confidence=confidence,
                           filename=filename,
                           heatmap=heatmap_filename,
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

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)