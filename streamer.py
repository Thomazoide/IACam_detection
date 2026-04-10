from flask import Flask, Response
import threading
import cv2
from os import getenv

class Streamer:
    def __init__(self):
        self.app = Flask(__name__)
        self.frame = None
        self.lock = threading.Lock()
        self.port = getenv("WORKER_PORT")
        @self.app.route("/stream")
        def stream():
            return Response(self.generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
        threading.Thread(target=self.run, daemon=True).start()
    def update(self, frame):
        with self.lock:
            self.frame = frame.copy()
    def generate(self):
        while True:
            with self.lock:
                if self.frame is None:
                    continue
                _, jpeg = cv2.imencode(".jpg", self.frame)
                frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    def run(self):
        self.app.run(host="0.0.0.0", port=self.port, debug=False)