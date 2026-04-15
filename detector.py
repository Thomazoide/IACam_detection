from ultralytics import YOLO
import cv2

model = YOLO("./yolov8n.pt")

def detect_people(frame):
    results = model.track(frame, persist=True)
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                detections.append({
                    "persona": True
                })
    return detections, frame