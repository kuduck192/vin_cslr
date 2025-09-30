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
        # Try different backends for cross-platform compatibility
        backends = [cv2.CAP_ANY, cv2.CAP_AVFOUNDATION]  # macOS compatible

        for backend in backends:
            try:
                self.cap = cv2.VideoCapture(0, backend)
                if self.cap.isOpened():
                    # Test if we can read a frame
                    ret, frame = self.cap.read()
                    if ret:
                        print(f"[CameraWorker] Camera initialized with backend: {backend}")
                        # Set some basic properties
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        return
                    else:
                        self.cap.release()
                        self.cap = None
            except Exception as e:
                print(f"[CameraWorker] Failed to initialize camera with backend {backend}: {e}")
                if self.cap:
                    self.cap.release()
                self.cap = None

        if not self.cap or not self.cap.isOpened():
            print("[CameraWorker] ERROR: Could not initialize camera with any backend")
            raise RuntimeError("Camera initialization failed")
    
    def run(self):
        if not self.cap or not self.cap.isOpened():
            print("[CameraWorker] ERROR: Camera not initialized, cannot start")
            return

        print("[CameraWorker] Starting camera capture loop")
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

            # Store the frame for UI display in main thread

        if frames:
            self.video_queue.put(frames)

    def close(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("[CameraWorker] Camera released and windows destroyed")