import cv2
import time
import queue
from workers.worker_manager import WorkerManager
from recognition_result import RecognitionResult
from collections import deque

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
        """Handle keyboard input using OpenCV waitKey (macOS compatible)"""
        key = cv2.waitKey(1) & 0xFF

        # Quit
        if key == ord('q') or key == 27:  # q or ESC
            return False

        # Toggle info panel
        if key == ord('i'):
            self.show_info_panel = not self.show_info_panel
            print(f"Info panel: {'ON' if self.show_info_panel else 'OFF'}")

        # Toggle debug info
        if key == ord('d'):
            self.show_debug_info = not self.show_debug_info
            print(f"Debug info: {'ON' if self.show_debug_info else 'OFF'}")

        # Pause/resume
        if key == ord(' '):  # space
            self.paused = not self.paused
            print(f"{'PAUSED' if self.paused else 'RESUMED'}")

        # Clear recognition history
        if key == ord('r'):
            self.recognition_history.clear()
            print("History cleared")

        # Increase/decrease frame skip
        if key == ord('+') or key == ord('='):
            self.frame_skip = min(30, self.frame_skip + 5)
            print(f"Frame skip: {self.frame_skip}")

        if key == ord('-'):
            self.frame_skip = max(1, self.frame_skip - 5)
            print(f"Frame skip: {self.frame_skip}")

        return True

    def run(self):
        self.initialize()
        self.worker_manager.run()
        prev_time = time.time()

        print("CSLR Demo started. Press 'q' or ESC to quit.")
        print("Controls: i=info panel, d=debug, space=pause, r=reset, +/-=frame skip")

        # Initialize a separate camera for UI display
        display_cap = cv2.VideoCapture(0)

        while True:

            self.check_recognition_results()

            # Display camera feed in main thread
            ret, frame = display_cap.read()
            if ret:
                frame = cv2.flip(frame, 1)  # Mirror effect

                # Draw UI on frame
                from ui.ui import draw_ui
                frame_with_ui = draw_ui(frame.copy(), self)
                cv2.imshow('CSLR Demo - Camera Feed', frame_with_ui)

            now = time.time()
            delta = now - prev_time
            if delta > 0:
                fps = 1 / delta
            else:
                fps = 0
            prev_time = now
            self.performance_stats['fps'] = fps

            if not self.handle_keyboard_input():
                display_cap.release()
                self.close()
                break

    def close(self):
        self.worker_manager.close()
        cv2.destroyAllWindows()
        print("[CSLRDemo] Demo ended")
