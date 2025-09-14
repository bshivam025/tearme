from ultralytics import YOLO
import cv2

# Load pretrained face detection model
model = YOLO("best.pt")  # or "yolov8n-face.pt"

# Read sample image
img_path = "upload/naukri.jpg"
img = cv2.imread(img_path)

# Run YOLO inference
results = model(img)

# Check if any faces detected
if len(results[0].boxes) == 0:
    print("No faces detected")
else:
    # Take first face
    box = results[0].boxes.xyxy[0]
    x1, y1, x2, y2 = map(int, box)

    # Crop face
    face_img = img[y1:y2, x1:x2]

    # Detect eyes inside face
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(face_img)

    # Draw blue tears under each eye
    for (ex, ey, ew, eh) in eyes:
        start = (x1 + ex + ew//2, y1 + ey + eh)
        end = (start[0], start[1] + int(0.15*(y2 - y1)))
        cv2.line(img, start, end, (255,0,0), 5)

# Save the processed image
cv2.imwrite("upload/naukri_tears.jpg", img)
print("Processed image saved as naukri_tears.jpg")
