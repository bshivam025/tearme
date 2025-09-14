from flask import Flask, request, render_template, send_file
from ultralytics import YOLO
import cv2
import os

# Load YOLO model once
model = YOLO("best.pt")  # or "yolov8n-face.pt"

# Path to save uploads
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def register_routes(app):

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/upload", methods=["POST"])
    def upload():
        if "file" not in request.files:
            return "No file part", 400

        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400

        # Save uploaded file
        save_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(save_path)

        # Read image
        img = cv2.imread(save_path)

        # Run YOLO detection
        results = model(img)

        # Process first face if detected
        if len(results[0].boxes) > 0:
            box = results[0].boxes.xyxy[0]
            x1, y1, x2, y2 = map(int, box)
            face_img = img[y1:y2, x1:x2]

            # Detect eyes with Haar cascade
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            eyes = eye_cascade.detectMultiScale(face_img)

            # Draw blue tears under eyes
            for (ex, ey, ew, eh) in eyes:
                start = (x1 + ex + ew//2, y1 + ey + eh)
                end = (start[0], start[1] + int(0.15*(y2 - y1)))
                cv2.line(img, start, end, (255,0,0), 5)

        # Save processed image
        output_path = os.path.join(UPLOAD_FOLDER, "processed_" + file.filename)
        cv2.imwrite(output_path, img)

        # Return processed image to frontend
        return send_file(output_path, mimetype='image/jpeg')
