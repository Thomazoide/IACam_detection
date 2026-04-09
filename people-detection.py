import cv2
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

rtsp_url = "rtsp://admin:Pr0liant@192.168.20.237:554/stream1"
cap = cv2.VideoCapture(rtsp_url)

LINE_Y = 300

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1,y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, "Persona", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0),2)
    reframedDetection = cv2.resize(frame, (1280, 720))
    cv2.imshow("Detección", reframedDetection)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()