import threading
import queue
import time
from collections import deque
from slr.slr import SLRModule
from recognition_result import RecognitionResult


class SLRWorker(threading.Thread):
    """Dedicated worker thread for Sign Language Recognition"""
    
    def __init__(self, video_queue: queue.Queue, result_queue: queue.Queue, stop_event: threading.Event,
                 device: str = 'cpu', model_name: str = 'corrnet', **slr_kwargs):
        super().__init__(daemon=True)
        self.video_queue = video_queue
        self.result_queue = result_queue

        # SLR processing parameters
        self.processing_interval = slr_kwargs.get('processing_interval', 0.5)
        self.last_process_time = 0
        self.stop_event = stop_event

        self.model_name = model_name
        self.device = device
        self.model = None
        

    def initialize(self):
        """Initialize the model for SLR."""
        self.model = SLRModule(model_name=self.model_name, device=self.device)

    def run(self):
        """Process frames from the queue for sign language recognition."""
        print("[SLR Worker] Started")

        while not self.stop_event.is_set():
            try:
                if not self.video_queue.empty():
                    frames = self.video_queue.get()
                    recognition_result = self.model.sign_language_recognition(frames)

                    recognition_result = RecognitionResult(
                        text=recognition_result.get('text', ''),
                        confidence=recognition_result.get('confidence', 0.0),
                        detected=recognition_result.get('detected', False),
                        metadata=recognition_result.get('metadata', { }),
                    )

                    print(recognition_result)
                    self.result_queue.put(recognition_result)
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"[SLR Worker] Error: {e}")
                raise e
            
    def close(self):
        print("[SLRWorker] SLRWorker closed")