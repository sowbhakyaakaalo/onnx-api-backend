from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import onnxruntime as ort
import cv2
from io import BytesIO
import logging

# ----------------------
# Logging for debugging
# ----------------------
logging.basicConfig(level=logging.INFO)

# ----------------------
# FastAPI setup
# ----------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------
# Load ONNX model
# ----------------------
session = ort.InferenceSession("model_- 21 april 2025 15_58.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Load class names
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ----------------------
# Prediction endpoint
# ----------------------
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    # Optional: Limit file size to 3MB
    if len(contents) > 3 * 1024 * 1024:
        logging.warning("File too large")
        return {"error": "File too large"}

    logging.info(f"Received file: {file.filename}")

    # Decode image
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    orig = img.copy()
    orig_h, orig_w = orig.shape[:2]

    # Preprocess image for model input
    img_resized = cv2.resize(img, (640, 640))
    img_input = img_resized.astype(np.float32)
    img_input = np.transpose(img_input, (2, 0, 1)) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    # Inference
    outputs = session.run([output_name], {input_name: img_input})
    predictions = outputs[0]

    if predictions.shape[1] < predictions.shape[2]:
        predictions = predictions.transpose(0, 2, 1)
    predictions = predictions[0]

    # ----------------------
    # Parse model predictions
    # ----------------------
    boxes, scores, class_ids = [], [], []
    threshold = 0.4

    for det in predictions:
        cx, cy, w, h = det[:4]
        class_scores = det[4:]
        score = np.max(class_scores)
        class_id = np.argmax(class_scores)

        if score > threshold:
            x1 = int((cx - w / 2) / 640 * orig_w)
            y1 = int((cy - h / 2) / 640 * orig_h)
            x2 = int((cx + w / 2) / 640 * orig_w)
            y2 = int((cy + h / 2) / 640 * orig_h)

            boxes.append([x1, y1, x2, y2])
            scores.append(float(score))
            class_ids.append(class_id)

    # ----------------------
    # Draw results
    # ----------------------
    if not boxes:
        # No detections: show label
        cv2.putText(orig, "NO DIMM DETECTED", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        logging.info("No DIMMs detected.")
    else:
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            class_name = class_names[class_ids[i]]
            conf = scores[i]
            label = f"{class_name} ({conf:.2f})"

            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(orig, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), (0, 255, 0), -1)
            # Draw label text
            cv2.putText(orig, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            # Draw bounding box
            cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)

        logging.info(f"Detected {len(boxes)} object(s).")

    # ----------------------
    # Encode and return image
    # ----------------------
    _, img_encoded = cv2.imencode(".jpg", orig)
    return StreamingResponse(BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

