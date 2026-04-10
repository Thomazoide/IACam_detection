from os import getenv
import requests
import cv2
from detector import detect_people
from streamer import Streamer

RTSP_URL = getenv("RTSP_URL")
CAMERA_ID = getenv("CAMERA_ID")
API_URL = getenv("API_URL")

streamer = Streamer()

cap = cv2.VideoCapture(RTSP_URL)


if not cap.isOpened():
    print("Error al conectar cámara")
    exit(1)

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Intentando reconectar")
        cap.release()
        cap = cv2.VideoCapture(RTSP_URL)
        continue
    frame_count += 1
    if frame_count%3 != 0:
        streamer.update(frame)
        continue
    detections, frame = detect_people(frame)
    for det in detections:
        if det["persona"]:
            try:
                requests.post(API_URL, json={
                    "camera_id": CAMERA_ID,
                    "event": "posible_intruso"
                }, timeout=1)
            except:
                pass
    streamer.update(frame)