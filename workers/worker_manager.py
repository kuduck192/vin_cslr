from camera_worker import CameraWorker
from cslr_worker import SLRWorker
import queue
import threading
import time

class WorkerManager:
    def __init__(self, video_queue: queue.Queue, result_queue: queue.Queue, model_name='corrnet', device='cpu'):
        self.video_queue = video_queue
        self.result_queue = result_queue
        self.stop_event = threading.Event()
        self.camera_worker = CameraWorker(self.video_queue, self.stop_event)
        self.slr_worker = SLRWorker(self.video_queue, self.result_queue, self.stop_event,
                                    model_name=model_name, device=device)

    def initialize(self):
        self.camera_worker.initialize()
        self.slr_worker.initialize()

    def run(self):
        self.camera_worker.start()
        self.slr_worker.start()
        print("[WorkerManager] Workers started")

    def close(self):
        self.stop_event.set()
        self.camera_worker.join()
        self.slr_worker.join()
        self.camera_worker.close()
        self.slr_worker.close()
        print("[WorkerManager] Workers stopped")
