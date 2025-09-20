import threading
import queue
import time
import cv2
from ui.ui import draw_ui
VIDEO_LENGTH_SEC = 2
FRAME_SIZE = (640, 480)

class CameraWorker(threading.Thread):
    def __init__(self, video_queue: queue.Queue, result_queue: queue.Queue, stop_event: threading.Event, demo=None):
        super().__init__(daemon=True)
        self.video_queue = video_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.demo = demo
        self.cap = None

    def initialize(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        print("[CameraWorker] Camera initialized")
    
    def run(self):
        while not self.stop_event.is_set():
            frames = []
            self._run_by_timer(frames=frames, video_length_sec=VIDEO_LENGTH_SEC)
    

    def _run_by_timer(self, frames, video_length_sec):
        """
        Capture video frames for a specific duration.
        Draw UI elements on the frames.
        """
        start_time = time.time()

        while time.time() - start_time < video_length_sec:
            ret, frame = self.cap.read()

            if not ret:
                break

            frame = cv2.resize(frame, FRAME_SIZE)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if self.demo:
                frame_ui = draw_ui(frame.copy(), self.demo)
            else:
                frame_ui = frame.copy()

            cv2.imshow('Camera', frame_ui)
            cv2.waitKey(1)

        if frames:
            self.video_queue.put(frames)

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print("[CameraWorker] Camera released and windows destroyed")