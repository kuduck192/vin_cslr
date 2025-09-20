import cv2
import time
import queue
from workers.worker_manager import WorkerManager
from recognition_result import RecognitionResult
from collections import deque
import keyboard

class CSLRDemo:
    def __init__(self, model_name='corrnet', device='cpu'):
        self.frame_queue = queue.Queue(maxsize=500)
        self.result_queue = queue.Queue(maxsize=50)
        self.queue_sizes = {
            'frame': 0,
            'result': 0,
            'tts': 0
        }


        self.worker_manager = WorkerManager(
            self.frame_queue, 
            self.result_queue,
            model_name=model_name, 
            device=device, 
            demo=self)

        # Initialize recognition history and performance stats
        self.recognition_history = deque(maxlen=5)
        self.current_result: RecognitionResult = None
        self.performance_stats = {'frames_processed':0,
                                  'recognitions_made':0,
                                  'tts_triggered':0,
                                  'start_time':time.time(),
                                  'fps': 0}
        
        # Demo settings
        self.show_info_panel = True
        self.show_debug_info = False
        self.frame_skip = 1
        
        self.paused = False

    def initialize(self):
        self.worker_manager.initialize()
        print("[CSLRDemo] Demo initialized")


    def check_recognition_results(self):
        """Check and process recognition results from the result queue."""

        while not self.result_queue.empty():
            result: RecognitionResult = self.result_queue.get_nowait()
            self.recognition_history.append({
                'text': result.text,
                'confidence': result.confidence,
                'timestamp': time.strftime("%H:%M:%S"),
                'metadata': result.metadata
            })
            self.current_result = result
            self.performance_stats['recognitions_made'] += 1

            self.queue_sizes['frame'] = self.frame_queue.qsize()
            self.queue_sizes['result'] = self.result_queue.qsize()
            # self.queue_sizes['tts'] = self.tts_queue.qsize()

    def handle_keyboard_input(self):
        """Non-blocking keyboard input using keyboard module"""
        # Quit
        if keyboard.is_pressed('q') or keyboard.is_pressed('esc'):
            return False

        # Toggle info panel
        if keyboard.is_pressed('i'):
            self.show_info_panel = not self.show_info_panel
            time.sleep(0.2)  

        # Toggle debug info
        if keyboard.is_pressed('d'):
            self.show_debug_info = not self.show_debug_info
            time.sleep(0.2)

        # Pause/resume
        if keyboard.is_pressed('space'):
            self.paused = not self.paused
            time.sleep(0.2)

        # Clear recognition history
        if keyboard.is_pressed('r'):
            self.recognition_history.clear()
            time.sleep(0.2)

        # Increase/decrease frame skip
        if keyboard.is_pressed('+') or keyboard.is_pressed('='):
            self.frame_skip = min(30, self.frame_skip + 5)
            time.sleep(0.2)

        if keyboard.is_pressed('-'):
            self.frame_skip = max(1, self.frame_skip - 5)
            time.sleep(0.2)

        return True

    def run(self):
        self.initialize()
        self.worker_manager.run()
        prev_time = time.time()

        while True:
            
            self.check_recognition_results()

            now = time.time()
            delta = now - prev_time
            if delta > 0:
                fps = 1 / delta
            else:
                fps = 0 
            prev_time = now
            self.performance_stats['fps'] = fps

            if not self.handle_keyboard_input():
                self.close()
                break
            
            time.sleep(0.01)

    def close(self):
        self.worker_manager.close()
        cv2.destroyAllWindows()
        print("[CSLRDemo] Demo ended")
