from flask import Flask, render_template, request, send_file, redirect, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os, sqlite3, datetime, time
import cv2
import numpy as np

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

    c.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        prediction TEXT,
        corrected TEXT,
        confidence REAL,
        timestamp TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ---------------- STATS ----------------
def get_stats():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM history")
    total_scans = c.fetchone()[0]

    c.execute("SELECT AVG(confidence) FROM history")
    avg_conf = c.fetchone()[0] or 0

    conn.close()

    return {
        "total_scans": total_scans,
        "accuracy": round(avg_conf, 2)
    }

# ---------------- CLASSES ----------------
classes = torch.load("classes.pth")

# ---------------- MODEL ----------------
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("waste_classifier_best.pth", map_location="cpu"))
model.eval()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
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

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

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

    h1.remove()
    h2.remove()

    return cam

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    stats = get_stats()
    return render_template("index.html", stats=stats)

@app.route("/upload")
def upload():
    return render_template("upload.html")

# ---------------- PREDICT ----------------
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

    # risk
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

    # ---------------- HEATMAP FIX ----------------
    cam = generate_heatmap(model, img_tensor, pred.item())

    cam = np.maximum(cam, 0)
    if cam.max() != 0:
        cam = cam / cam.max()

    cam = cv2.GaussianBlur(cam, (51, 51), 0)
    cam = np.power(cam, 1.5)

    heat = np.uint8(255 * cam)
    heatmap_color = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)

    glow = cv2.GaussianBlur(heatmap_color, (35, 35), 0)
    heatmap_color = cv2.addWeighted(heatmap_color, 0.7, glow, 0.3, 0)

    original = cv2.imread(filepath)
    original = cv2.resize(original, (224, 224))

    overlay = cv2.addWeighted(original, 0.55, heatmap_color, 0.65, 0)

    heatmap_filename = "heatmap_" + filename
    heatmap_path = os.path.join("static/uploads", heatmap_filename)
    cv2.imwrite(heatmap_path, overlay)

    # save DB
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO history (filename, prediction, confidence, timestamp)
        VALUES (?, ?, ?, ?)
    """, (filename, prediction, confidence, str(datetime.datetime.now())))
    conn.commit()
    conn.close()

    return render_template(
        "result.html",
        prediction=prediction,
        confidence=confidence,
        filename=filename,
        heatmap=heatmap_filename,
        risk=risk,
        instruction=instruction
    )

# ---------------- DASHBOARD ----------------
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

    last_updated = logs[0][4] if logs else None

    return render_template(
        "dashboard.html",
        total=total,
        data=data,
        last_updated=last_updated,
        logs=logs
    )

# ---------------- EXTRA PAGES ----------------
@app.route("/who-guidelines")
def who():
    return render_template("who.html")

@app.route("/biomedical-rules")
def biomedical():
    return render_template("biomedical.html")

@app.route("/privacy-policy")
def privacy():
    return render_template("privacy.html")

@app.route("/terms")
def terms():
    return render_template("terms.html")

@app.route("/delete/<int:id>")
def delete(id):
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("DELETE FROM history WHERE id=?", (id,))
    conn.commit()
    conn.close()
    return redirect("/dashboard")

@app.route("/export-pdf")
def export_pdf():
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Image as RLImage, Spacer
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    import os
    import sqlite3

    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT filename, prediction, confidence, timestamp FROM history")
    data = c.fetchall()
    conn.close()

    file_path = "static/report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=letter)

    styles = getSampleStyleSheet()

    # custom style for table text wrapping
    cell_style = ParagraphStyle(
        'cell',
        fontSize=8,
        leading=10,
        wordWrap='CJK',  # important for wrapping
    )

    title = Paragraph("SafeWaste AI - Scan Report", styles["Title"])

    elements = []
    elements.append(title)
    elements.append(Spacer(1, 12))

    table_data = [
        [
            "Image",
            Paragraph("Filename", cell_style),
            Paragraph("Prediction", cell_style),
            Paragraph("Confidence", cell_style),
            Paragraph("Timestamp", cell_style)
        ]
    ]

    for row in data:
        filename, prediction, confidence, timestamp = row

        img_path = os.path.join("static/uploads", filename)

        if os.path.exists(img_path):
            img = RLImage(img_path, width=40, height=40)
        else:
            img = Paragraph("No Image", cell_style)

        table_data.append([
            img,
            Paragraph(str(filename), cell_style),
            Paragraph(str(prediction), cell_style),
            Paragraph(f"{confidence}%", cell_style),
            Paragraph(str(timestamp), cell_style)
        ])

    # better balanced column widths
    table = Table(table_data, colWidths=[60, 120, 120, 80, 140], repeatRows=1)

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),

        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),

        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))

    elements.append(table)

    doc.build(elements)

    return send_file(file_path, as_attachment=True)

@app.route("/delete-multiple", methods=["POST"])
def delete_multiple():
    data = request.get_json()
    ids = data.get("ids", [])

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.executemany("DELETE FROM history WHERE id=?", [(i,) for i in ids])

    conn.commit()
    conn.close()

    return jsonify({"status": "success"})

@app.route("/feedback", methods=["POST"])
def feedback():
    image = request.form.get("image")
    prediction = request.form.get("prediction")
    confidence = request.form.get("confidence")
    corrected = request.form.get("correct_label")

    conn = sqlite3.connect("database.db")
    c = conn.cursor()

    c.execute("""
        INSERT INTO feedback (filename, prediction, corrected, confidence, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (image, prediction, corrected, confidence, str(datetime.datetime.now())))

    conn.commit()
    conn.close()

    return redirect("/dashboard?msg=correction_saved")

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)