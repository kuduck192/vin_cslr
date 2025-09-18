import threading
import queue
import time
import cv2

VIDEO_LENGTH_SEC = 2
FRAME_SIZE = (640, 480)

class CameraWorker(threading.Thread):
    def __init__(self, video_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.video_queue = video_queue
        self.stop_event = stop_event
        self.cap = None

    def initialize(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        print("[CameraWorker] Camera initialized")
    
    def run(self):
        while not self.stop_event.is_set():
            frames = []
            self._run_by_timer(frames=frames, video_length_sec=VIDEO_LENGTH_SEC)
    

    def _run_by_timer(self, frames, video_length_sec):
        start_time = time.time()
        while time.time() - start_time < video_length_sec:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, FRAME_SIZE)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cv2.imshow('Camera', frame)
            cv2.waitKey(1)
            # if cv2.waitKey(1) & 0xFF == ord('b'):
            #     self.stop_event.set()
            #     break
        if frames:
            self.video_queue.put(frames)

    def close(self):
        print("[CameraWorker] Closing CameraWorker")
        self.cap.release()
        cv2.destroyAllWindows()
        print("[CameraWorker] Camera released and windows destroyed")