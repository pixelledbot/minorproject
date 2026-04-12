from flask import Flask, render_template, request, send_file
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os, sqlite3, datetime, time
import cv2
import numpy as np

# PDF
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image as RLImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

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

# ---------------- LOAD CLASSES ----------------
classes = torch.load("classes.pth")
print("Loaded classes:", classes)

# ---------------- MODEL ----------------
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("waste_classifier_best.pth", map_location="cpu"))
model.eval()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------- GRAD-CAM ----------------
def generate_heatmap(model, image_tensor, class_idx):
    gradients = []
    activations = []

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

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

    cam = cv2.resize(cam, (224, 224))

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

    os.makedirs("static/uploads", exist_ok=True)
    filename = str(int(time.time())) + "_" + file.filename
    filepath = os.path.join("static/uploads", filename)
    file.save(filepath)

    img = Image.open(filepath).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    outputs = model(img_tensor)
    probs = torch.softmax(outputs, dim=1)
    conf, pred = torch.max(probs, 1)

    prediction = classes[pred.item()]
    confidence = round(conf.item() * 100, 2)

    all_probs = {
        classes[i]: round(probs[0][i].item() * 100, 2)
        for i in range(len(classes))
    }

    if prediction in ["infectious", "sharps"]:
        risk = "High"
    elif prediction == "pharmaceutical":
        risk = "Medium"
    else:
        risk = "Low"

    instructions = {
        "general": "Dispose in green bin.",
        "infectious": "Dispose in yellow biohazard bag.",
        "pharmaceutical": "Dispose in blue chemical bin.",
        "sharps": "Dispose in puncture-proof container."
    }

    instruction = instructions.get(prediction, "Dispose properly.")

    # HEATMAP
    heatmap = generate_heatmap(model, img_tensor, pred.item())

    original = cv2.imread(filepath)
    original = cv2.resize(original, (224, 224))

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
        (filename, prediction, confidence, str(datetime.datetime.now()))
    )
    conn.commit()
    conn.close()

    return render_template("result.html",
                           prediction=prediction,
                           confidence=confidence,
                           filename=filename,
                           heatmap=heatmap_filename,
                           risk=risk,
                           instruction=instruction,
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

# ---------------- PDF WITH IMAGES ----------------
@app.route("/download_pdf")
def download_pdf():

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT * FROM history")
    data = c.fetchall()
    conn.close()

    filename = "history.pdf"
    pdf = SimpleDocTemplate(filename, pagesize=letter)

    elements = []
    style = getSampleStyleSheet()

    elements.append(Paragraph("Waste Classification History", style['Title']))

    table_data = [["Image", "Prediction", "Confidence", "Timestamp"]]

    for row in data:
        file_name = row[1]
        prediction = row[2]
        confidence = str(row[3]) + "%"
        timestamp = row[4]

        image_path = os.path.join("static/uploads", file_name)

        if os.path.exists(image_path):
            img = RLImage(image_path, width=90, height=90)
        else:
            img = "No Image"

        table_data.append([img, prediction, confidence, timestamp])

    table = Table(table_data, colWidths=[100, 120, 80, 180])

    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
    ]))

    elements.append(table)
    pdf.build(elements)

    return send_file(filename, as_attachment=True)

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)