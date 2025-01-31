import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from pydantic import BaseModel
from typing import List

# Initialize FastAPI app
app = FastAPI()

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Ensure you have this file in the same directory

# Define response model
class DetectionResult(BaseModel):
    object: str
    position: str
    command: str

# Define function to determine object position
def get_position_and_command(x, frame_width):
    if x < frame_width // 3:
        return "Left", "Move right"
    elif x > 2 * frame_width // 3:
        return "Right", "Move left"
    else:
        return "Center", "Stop and assess"

@app.post("/detect", response_model=List[DetectionResult])
async def detect_objects(file: UploadFile = File(...)):
    # Convert uploaded file to an OpenCV image
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Get frame dimensions
    frame_height, frame_width, _ = frame.shape

    # Perform object detection
    results = model(frame)

    detections = []
    for detection in results[0].boxes.data:
        x1, y1, x2, y2 = map(int, detection[:4])  # Bounding box
        obj_class = int(detection[5])  # Object class index
        object_name = model.names[obj_class]

        # Determine position
        object_x_center = (x1 + x2) // 2
        position, command = get_position_and_command(object_x_center, frame_width)

        detections.append(DetectionResult(object=object_name, position=position, command=command))

    return detections
