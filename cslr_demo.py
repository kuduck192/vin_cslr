import cv2
import time
import queue
from worker_manager import WorkerManager
from recognition_result import RecognitionResult
from ui import draw_ui
from collections import deque
import keyboard

class CSLRDemo:
    def __init__(self, model_name='corrnet', device='cpu'):
        self.frame_queue = queue.Queue(maxsize=500)
        self.result_queue = queue.Queue(maxsize=50)
        self.worker_manager = WorkerManager(self.frame_queue, self.result_queue,
                                            model_name=model_name, device=device)

        self.recognition_history = deque(maxlen=5)
        self.performance_stats = {'frames_processed':0,'recognitions_made':0,'tts_triggered':0,'start_time':time.time()}
        self.show_info_panel = True
        self.show_debug_info = False
        self.frame_counter = 0
        self.frame_skip = 1
        self.current_result: RecognitionResult = None
        self.paused = False

    def initialize(self):
        self.worker_manager.initialize()
        print("[CSLRDemo] Demo initialized")

    def process_frame_async(self, frame):
        self.frame_counter += 1
        if self.frame_counter % self.frame_skip == 0:
            try:
                self.frame_queue.put_nowait(frame)
                self.performance_stats['frames_processed'] += 1
            except queue.Full:
                pass

    def check_recognition_results(self):
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

    def handle_keyboard_input(self, key):
        if key in ['q', 'esc']:
            return False
        elif key == ord('i'):
            self.show_info_panel = not self.show_info_panel
        elif key == ord('d'):
            self.show_debug_info = not self.show_debug_info
        elif key == ord(' '):
            self.paused = not self.paused
        elif key == ord('r'):
            self.recognition_history.clear()
        elif key == ord('+') or key == ord('='):
            self.frame_skip = min(30, self.frame_skip + 5)
        elif key == ord('-'):
            self.frame_skip = max(1, self.frame_skip - 5)
        else:
            return True

    def run(self):
        self.initialize()
        self.worker_manager.run()
        time.sleep(0.5)
        prev_time = time.time()

        # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        while True:
            # try:
            #     frame = self.frame_queue.get(timeout=1.0)  # frame mới nhất từ CameraWorker
            # except queue.Empty:
            #     continue

            # if not self.paused:
            #     self.process_frame_async(frame)
            #     self.check_recognition_results()

            # fps = 1 / (time.time() - prev_time)
            # prev_time = time.time()
            # self.performance_stats['fps'] = fps

            # frame = draw_ui(frame, self)
           
            if keyboard.read_event().event_type == keyboard.KEY_DOWN:
                key = keyboard.read_event().name
                print(f"Bạn đã nhấn phím: {key}")

            if not self.handle_keyboard_input(key):
                self.close()
                break


    def close(self):
        self.worker_manager.close()
        cv2.destroyAllWindows()
        print("[CSLRDemo] Demo ended")
