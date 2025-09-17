import time
import threading
import queue

# import for camera thread
import cv2
from config import config

class CameraWorker(threading.Thread):
    def __init__(self, video_queue: queue.Queue, stop_event: threading.Event, cap: cv2.VideoCapture):
        super().__init__(daemon=True)
        self.video_queue = video_queue
        self.stop_event = stop_event
        self.cap = cap

    def initialize(self):
        pass
    
    def run(self):
        while not self.stop_event.is_set():
            frames = []
            # change this method when we want to run by another, ie. put to queue when we have start and end signal of 1 action
            self._run_by_timer(frames=frames, video_length_sec=config.video_length_sec)
    
    def close(self):
        pass
    
    def _run_by_timer(self, frames, video_length_sec):
        start_time = time.time()
        while time.time() - start_time < video_length_sec:
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if frames:
            self.video_queue.put(frames)
