'''File containing worker thread for:
- CameraWorker: Camera Capture
- SLRWorker: Sign Language Recognition
- TTSWorker: Text to Speech'''

import threading
import queue

# import for SLR thread
from collections import deque
import time
from slr.slr import SLRModule

from config import config

class SLRWorker(threading.Thread):
    """Dedicated worker thread for Sign Language Recognition"""
    
    def __init__(self, video_queue: queue.Queue, result_queue: queue.Queue, stop_event: threading.Event,
                 device: str = 'cpu', model_name: str = 'corrnet',
                 **slr_kwargs):
        super().__init__(daemon=True)
        # this queue is shared with camera thread
        self.video_queue = video_queue
        # this queue is shared with tts thread
        self.result_queue = result_queue
        self.processing_interval = slr_kwargs.get('processing_interval', 0.5)
        self.last_process_time = 0
        
        self.window_size_frames = slr_kwargs.get('window_size_frames', 60)
        self.frame_buffer = deque(maxlen=self.window_size_frames)
        self.stop_event = stop_event
        self.model_name = model_name
        self.device = device
        self.model = None
        self.initialize()

    def initialize(self):
        if self.model is None:
            self.model = SLRModule(model_name=self.model_name, device=self.device)

    def run(self):
        '''When thread starts, run this method'''
        while not self.stop_event.is_set():
            try:
                if self.video_queue:
                    # get video from video queue
                    frames = self.video_queue.get()

                    # Sanity check
                    if frames is None: continue
                    if len(frames) > config.min_frame_to_infer and frames[0].ndim != 3: continue # Invalid size
                    
                    recognition_result = self.model.sign_language_recognition(frames)
                    print(recognition_result)
                    # append to queue for tts worker
                    self.result_queue.put(recognition_result)
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"[SLR Worker] Error: {e}")
                raise e
    
    def close(self):
        pass