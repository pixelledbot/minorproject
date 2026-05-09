from flask import Flask, render_template, request, send_file, redirect, jsonify
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import sqlite3
import datetime
import time
import cv2
import numpy as np

from flask_mail import Mail, Message

from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    Image as RLImage
)

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# =========================
# APP
# =========================

app = Flask(__name__)

# =========================
# EMAIL CONFIG
# =========================

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False

app.config['MAIL_USERNAME'] = 'simran2315222@gmail.com'

# IMPORTANT → KEEP SPACES
app.config['MAIL_PASSWORD'] = 'ztep dess iios guid'

app.config['MAIL_DEFAULT_SENDER'] = 'simran2315222@gmail.com'

mail = Mail(app)

# =========================
# DATABASE
# =========================

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

# =========================
# STATS
# =========================

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

# =========================
# LOAD MODEL
# =========================

classes = torch.load("classes.pth")

model = models.resnet18(pretrained=False)

model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)

model.load_state_dict(
    torch.load(
        "waste_classifier_best.pth",
        map_location="cpu"
    )
)

model.eval()

# =========================
# IMAGE TRANSFORM
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# =========================
# GRAD CAM
# =========================

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

# =========================
# ROUTES
# =========================

@app.route("/")
def home():

    stats = get_stats()

    return render_template(
        "home.html",
        stats=stats
    )

@app.route("/upload")
def upload():
    return render_template("upload.html")

# =========================
# PREDICT
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files.get("image")

    if file is None or file.filename == "":
        return "No file uploaded"

    os.makedirs("static/uploads", exist_ok=True)

    filename = str(int(time.time())) + "_" + file.filename

    filepath = os.path.join(
        "static/uploads",
        filename
    )

    file.save(filepath)

    img = Image.open(filepath).convert("RGB")

    img_tensor = transform(img).unsqueeze(0)

    outputs = model(img_tensor)

    probs = torch.softmax(outputs, dim=1)

    conf, pred = torch.max(probs, 1)

    prediction = classes[pred.item()]

    confidence = round(conf.item() * 100, 2)

    if prediction in ["infectious", "sharps"]:
        risk = "High"

    elif prediction == "pharmaceutical":
        risk = "Medium"

    else:
        risk = "Low"

    instructions = {
        "general": "Dispose in BLACK bin.",
        "infectious": "Dispose in YELLOW bin.",
        "pharmaceutical": "Dispose in BLUE bin.",
        "sharps": "Dispose in WHITE bin."
    }

    instruction = instructions.get(
        prediction,
        "Dispose properly."
    )

    # =========================
    # HEATMAP
    # =========================

    cam = generate_heatmap(
        model,
        img_tensor,
        pred.item()
    )

    cam = np.maximum(cam, 0)

    if cam.max() != 0:
        cam = cam / cam.max()

    cam = cv2.GaussianBlur(cam, (51, 51), 0)

    heat = np.uint8(255 * cam)

    heatmap_color = cv2.applyColorMap(
        heat,
        cv2.COLORMAP_TURBO
    )

    original = cv2.imread(filepath)

    original = cv2.resize(original, (224, 224))

    overlay = cv2.addWeighted(
        original,
        0.55,
        heatmap_color,
        0.65,
        0
    )

    heatmap_filename = "heatmap_" + filename

    heatmap_path = os.path.join(
        "static/uploads",
        heatmap_filename
    )

    cv2.imwrite(
        heatmap_path,
        overlay
    )

    # =========================
    # SAVE DATABASE
    # =========================

    conn = sqlite3.connect("database.db")

    c = conn.cursor()

    c.execute("""
        INSERT INTO history
        (filename, prediction, confidence, timestamp)
        VALUES (?, ?, ?, ?)
    """, (
        filename,
        prediction,
        confidence,
        str(datetime.datetime.now())
    ))

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

# =========================
# DASHBOARD
# =========================

@app.route("/dashboard")
def dashboard():

    conn = sqlite3.connect("database.db")

    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM history")
    total = c.fetchone()[0]

    c.execute("""
        SELECT prediction, COUNT(*)
        FROM history
        GROUP BY prediction
    """)

    data = c.fetchall()

    c.execute("""
        SELECT *
        FROM history
        ORDER BY id DESC
    """)

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

# =========================
# EXPORT PDF
# =========================

@app.route("/export-pdf")
def export_pdf():

    conn = sqlite3.connect("database.db")

    cursor = conn.cursor()

    cursor.execute("""
        SELECT id,
               filename,
               prediction,
               confidence,
               timestamp
        FROM history
        ORDER BY id DESC
    """)

    logs = cursor.fetchall()

    conn.close()

    # =========================
    # STATS
    # =========================

    total_predictions = len(logs)

    avg_conf = 0

    if total_predictions > 0:

        avg_conf = round(
            sum(log[3] for log in logs)
            / total_predictions,
            2
        )

    counts = {
        "general": 0,
        "infectious": 0,
        "pharmaceutical": 0,
        "sharps": 0
    }

    for log in logs:

        pred = log[2].lower()

        if pred in counts:
            counts[pred] += 1

    # =========================
    # PDF
    # =========================

    pdf_path = "safewaste_report.pdf"

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=20,
        leftMargin=20,
        topMargin=20,
        bottomMargin=20
    )

    styles = getSampleStyleSheet()

    elements = []

    # =========================
    # TITLE
    # =========================

    title_style = ParagraphStyle(
        'title',
        parent=styles['Heading1'],
        fontSize=24,
        leading=30,
        textColor=colors.darkgreen,
        spaceAfter=20
    )

    title = Paragraph(
        "SafeWaste AI - Prediction Report",
        title_style
    )

    elements.append(title)

    generated = Paragraph(
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles['BodyText']
    )

    elements.append(generated)

    elements.append(Spacer(1, 20))

    # =========================
    # SUMMARY TABLE
    # =========================

    summary_data = [

        ["Metric", "Value"],

        ["Total Predictions", str(total_predictions)],

        ["Average Confidence", f"{avg_conf}%"],

        ["Sharps Count", str(counts["sharps"])],

        ["General Count", str(counts["general"])],

        ["Pharmaceutical Count", str(counts["pharmaceutical"])],

        ["Infectious Count", str(counts["infectious"])]

    ]

    summary_table = Table(
        summary_data,
        colWidths=[220, 180]
    )

    summary_table.setStyle(TableStyle([

        ('BACKGROUND', (0,0), (-1,0), colors.darkgreen),

        ('TEXTCOLOR', (0,0), (-1,0), colors.white),

        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),

        ('GRID', (0,0), (-1,-1), 1, colors.black),

        ('BACKGROUND', (0,1), (-1,-1), colors.beige),

        ('BOTTOMPADDING', (0,0), (-1,0), 10),

    ]))

    elements.append(summary_table)

    elements.append(Spacer(1, 30))

    # =========================
    # HISTORY TITLE
    # =========================

    history_title = Paragraph(
        "Detailed Prediction History",
        styles['Heading2']
    )

    elements.append(history_title)

    elements.append(Spacer(1, 12))

    # =========================
    # HISTORY TABLE
    # =========================

    history_data = [[
        "Image",
        "Prediction",
        "Confidence",
        "Timestamp"
    ]]

    for log in logs:

        image_path = os.path.join(
            "static/uploads",
            log[1]
        )

        # =========================
        # IMAGE
        # =========================

        if os.path.exists(image_path):

            try:

                img = RLImage(
                    image_path,
                    width=55,
                    height=55
                )

            except:

                img = "No Image"

        else:

            img = "Missing"

        history_data.append([

            img,

            str(log[2]).capitalize(),

            f"{log[3]}%",

            str(log[4])[:16]

        ])

    history_table = Table(
        history_data,
        colWidths=[80, 120, 100, 180]
    )

    history_table.setStyle(TableStyle([

        ('BACKGROUND', (0,0), (-1,0), colors.darkgreen),

        ('TEXTCOLOR', (0,0), (-1,0), colors.white),

        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),

        ('GRID', (0,0), (-1,-1), 1, colors.black),

        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),

        ('ALIGN', (0,0), (-1,-1), 'CENTER'),

        ('BOTTOMPADDING', (0,0), (-1,0), 12),

        ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),

    ]))

    elements.append(history_table)

    # =========================
    # BUILD
    # =========================

    doc.build(elements)

    return send_file(
        pdf_path,
        as_attachment=True,
        download_name="SafeWaste_Report.pdf"
    )
# =========================
# SEND EMAIL
# =========================

@app.route("/send-dashboard-email")
def send_dashboard_email():

    try:

        print("STARTING EMAIL REPORT...")

        conn = sqlite3.connect("database.db")

        cursor = conn.cursor()

        cursor.execute("""
            SELECT id,
                   filename,
                   prediction,
                   confidence,
                   timestamp
            FROM history
            ORDER BY id DESC
        """)

        all_logs = cursor.fetchall()

        conn.close()

        # =========================
        # FULL DATABASE FOR STATS
        # =========================

        stats_logs = all_logs

        # =========================
        # ONLY LATEST 20 FOR TABLE
        # =========================

        logs = all_logs[:20]

        # =========================
        # STATS
        # =========================

        total_predictions = len(stats_logs)

        avg_conf = 0

        if total_predictions > 0:

            avg_conf = round(

                sum(log[3] for log in stats_logs)

                / total_predictions,

                2
            )

        counts = {
            "general": 0,
            "infectious": 0,
            "pharmaceutical": 0,
            "sharps": 0
        }

        for log in stats_logs:

            pred = log[2].lower()

            if pred in counts:
                counts[pred] += 1

        # =========================
        # PDF GENERATION
        # =========================

        pdf_path = "dashboard_report.pdf"

        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            rightMargin=20,
            leftMargin=20,
            topMargin=20,
            bottomMargin=20
        )

        styles = getSampleStyleSheet()

        elements = []

        # =========================
        # TITLE
        # =========================

        title_style = ParagraphStyle(
            'title',
            parent=styles['Heading1'],
            fontSize=24,
            leading=30,
            textColor=colors.darkgreen,
            spaceAfter=20
        )

        title = Paragraph(
            "SafeWaste AI - Prediction Report",
            title_style
        )

        elements.append(title)

        generated = Paragraph(
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles['BodyText']
        )

        elements.append(generated)

        elements.append(Spacer(1, 20))

        # =========================
        # SUMMARY TABLE
        # =========================

        summary_data = [

            ["Metric", "Value"],

            ["Total Predictions", str(total_predictions)],

            ["Average Confidence", f"{avg_conf}%"],

            ["Sharps Count", str(counts["sharps"])],

            ["General Count", str(counts["general"])],

            ["Pharmaceutical Count", str(counts["pharmaceutical"])],

            ["Infectious Count", str(counts["infectious"])]

        ]

        summary_table = Table(
            summary_data,
            colWidths=[220, 180]
        )

        summary_table.setStyle(TableStyle([

            ('BACKGROUND', (0,0), (-1,0), colors.darkgreen),

            ('TEXTCOLOR', (0,0), (-1,0), colors.white),

            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),

            ('GRID', (0,0), (-1,-1), 1, colors.black),

            ('BACKGROUND', (0,1), (-1,-1), colors.beige),

            ('BOTTOMPADDING', (0,0), (-1,0), 10),

        ]))

        elements.append(summary_table)

        elements.append(Spacer(1, 30))

        # =========================
        # HISTORY TITLE
        # =========================

        history_title = Paragraph(
            "Detailed Prediction History",
            styles['Heading2']
        )

        elements.append(history_title)

        elements.append(Spacer(1, 12))

        # =========================
        # HISTORY TABLE
        # =========================

        history_data = [[
            "Image",
            "Prediction",
            "Confidence",
            "Timestamp"
        ]]

        for log in logs:

            image_path = os.path.join(
                "static/uploads",
                log[1]
            )

            # =========================
            # IMAGE
            # =========================

            if os.path.exists(image_path):

                try:

                    img = RLImage(
                        image_path,
                        width=40,
                        height=40
                    )

                except:

                    img = "No Image"

            else:

                img = "Missing"

            history_data.append([

                img,

                str(log[2]).capitalize(),

                f"{log[3]}%",

                str(log[4])[:16]

            ])

        history_table = Table(
            history_data,
            colWidths=[70, 120, 90, 180]
        )

        history_table.setStyle(TableStyle([

            ('BACKGROUND', (0,0), (-1,0), colors.darkgreen),

            ('TEXTCOLOR', (0,0), (-1,0), colors.white),

            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),

            ('GRID', (0,0), (-1,-1), 1, colors.black),

            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),

            ('ALIGN', (0,0), (-1,-1), 'CENTER'),

            ('BOTTOMPADDING', (0,0), (-1,0), 12),

            ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke),

        ]))

        elements.append(history_table)

        # =========================
        # FOOTER NOTE
        # =========================

        elements.append(Spacer(1, 20))

        footer = Paragraph(
            "This report was generated automatically by SafeWaste AI.",
            styles['Italic']
        )

        elements.append(footer)

        # =========================
        # BUILD PDF
        # =========================

        print("BUILDING PDF...")

        doc.build(elements)

        print("PDF BUILT")

        print(
            "PDF SIZE:",
            round(
                os.path.getsize(pdf_path)
                / 1024 / 1024,
                2
            ),
            "MB"
        )

        # =========================
        # SEND EMAIL
        # =========================

        print("SENDING EMAIL...")

        msg = Message(
            subject="SafeWaste AI Dashboard Report",
            recipients=["simran2315222@gmail.com"]
        )

        msg.body = """
SafeWaste AI Report Attached.

The latest dashboard analytics report has been generated successfully.

Only latest 20 predictions are included in the PDF table, but overall stats are based on the entire database.
"""

        with open(pdf_path, "rb") as fp:

            msg.attach(
                "SafeWaste_Report.pdf",
                "application/pdf",
                fp.read()
            )

        mail.send(msg)

        print("EMAIL SENT SUCCESSFULLY")

        # =========================
        # CLEANUP
        # =========================

        if os.path.exists(pdf_path):

            os.remove(pdf_path)

        return "EMAIL SENT SUCCESSFULLY"

    except Exception as e:

        print("EMAIL ERROR:")
        print(str(e))

        return str(e)

# =========================
# DELETE MULTIPLE
# =========================

@app.route("/delete-multiple", methods=["POST"])
def delete_multiple():

    data = request.get_json()

    ids = data.get("ids", [])

    conn = sqlite3.connect("database.db")

    c = conn.cursor()

    c.executemany(
        "DELETE FROM history WHERE id=?",
        [(i,) for i in ids]
    )

    conn.commit()

    conn.close()

    return jsonify({
        "status": "success"
    })

# =========================
# FEEDBACK
# =========================

@app.route("/feedback", methods=["POST"])
def feedback():

    image = request.form.get("image")

    prediction = request.form.get("prediction")

    confidence = request.form.get("confidence")

    corrected = request.form.get("correct_label")

    conn = sqlite3.connect("database.db")

    c = conn.cursor()

    c.execute("""
        INSERT INTO feedback
        (filename, prediction, corrected,
         confidence, timestamp)

        VALUES (?, ?, ?, ?, ?)
    """, (
        image,
        prediction,
        corrected,
        confidence,
        str(datetime.datetime.now())
    ))

    conn.commit()

    conn.close()

    return redirect("/dashboard")

# =========================
# INFO PAGES
# =========================

@app.route("/who")
def who():
    return render_template("who.html")

@app.route("/biomedical")
def biomedical():
    return render_template("biomedical.html")

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

@app.route("/terms")
def terms():
    return render_template("terms.html")

# =========================
# RUN
# =========================

if __name__ == "__main__":

    app.run(
        debug=False,
        threaded=True
    )