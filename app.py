from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import onnxruntime as ort
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session = ort.InferenceSession("model_- 21 april 2025 15_58.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    orig_h, orig_w = img.shape[:2]
    img_resized = cv2.resize(img, (640, 640))
    img_input = img_resized.astype(np.float32)
    img_input = np.transpose(img_input, (2, 0, 1)) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    outputs = session.run([output_name], {input_name: img_input})
    predictions = outputs[0]

    if predictions.shape[1] < predictions.shape[2]:
        predictions = predictions.transpose(0, 2, 1)
    predictions = predictions[0]

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

    detections = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        class_name = class_names[class_ids[i]]
        conf = scores[i]
        detections.append({
            "box": [x1, y1, x2, y2],
            "class": class_name,
            "score": round(conf, 2)
        })

    return JSONResponse(content={"detections": detections})

