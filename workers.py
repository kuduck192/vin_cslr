'''File containing worker thread for:
- CameraWorker: Camera Capture
- SLRWorker: Sign Language Recognition
- TTSWorker: Text to Speech'''

import threading
import queue

# import for camera thread
import cv2
from config import FRAME_SIZE, VIDEO_LENGTH_SEC

# import for SLR thread
from collections import deque
import time
from slr.slr import SLRModule

class CameraWorker(threading.Thread):
    def __init__(self, video_queue: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.video_queue = video_queue
        self.stop_event = stop_event
        self.cap = cv2.VideoCapture(0)

    def init(self):
        pass
    
    def run(self):
        while not self.stop_event.is_set():
            frames = []
            # change this method when we want to run by another, ie. put to queue when we have start and end signal of 1 action
            self._run_by_timer(frames=frames, video_length_sec=VIDEO_LENGTH_SEC)
        self.cap.release()
        cv2.destroyAllWindows()
    
    def close(self):
        pass

    def _run_by_timer(self, frames, video_length_sec):
        start_time = time.time()
        while time.time() - start_time < video_length_sec:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, FRAME_SIZE)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cv2.imshow('Camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()
                break
        if frames:
            self.video_queue.put(frames)

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
        self.init()

    def init(self):
        self.model = SLRModule(model_name=self.model_name, device=self.device)

    def run(self):
        '''When thread starts, run this method'''
        while not self.stop_event.is_set():
            try:
                if self.video_queue:
                    # get video from video queue
                    frames = self.video_queue.get()
                    # in sign_language recognition module, do all the process and return right format
                    recognition_result = self.model.sign_language_recognition(frames)
                    print(recognition_result)
                    # append to queue for tts worker
                    self.result_queue.put(recognition_result)
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"[SLR Worker] Error: {e}")
                raise e


if __name__ == '__main__':
    video_queue = queue.Queue(maxsize=200)
    result_queue = queue.Queue(maxsize=50)

    stop_event = threading.Event()

    camera = CameraWorker(video_queue=video_queue, stop_event=stop_event)
    slr = SLRWorker(video_queue=video_queue, result_queue=result_queue, stop_event=stop_event, device='cpu', model_name='corrnet')

    camera.start()
    slr.start() 
    camera.join()
    slr.join()