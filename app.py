"""
Real-time Continuous Sign Language Recognition (CSLR) Demo
Video -> SLR -> TTS -> Audio Pipeline
Improved version with better threading architecture
"""
import cv2
import numpy as np
import time
from collections import deque
import threading
import queue
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from config import config

# Import custom modules
# from tts import text_to_speech_async

from workers import SLRWorker, CameraWorker, TTSWorker


@dataclass
class RecognitionResult:
    """Data class for recognition results"""
    detected: bool
    text: str
    confidence: float
    metadata: dict




class CSLRApp:
    def __init__(self):
        """Initialize CSLR demo application with improved threading"""
        # Video capture
        self.cap = None
        self.frame_width = config.frame_width
        self.frame_height = config.frame_height
        self.frame_counter = 0
        
        # Threading components
        self.video_queue = queue.Queue(maxsize=config.frame_queue_size)
        self.slr_queue = queue.Queue(maxsize=config.slr_queue_size)
        self.tts_queue = queue.Queue(maxsize=config.tts_queue_size)
        
        # Worker threads
        self.cam_worker = None
        self.slr_worker = None
        self.tts_worker = None
        self.stop_event = threading.Event()
        
        # Recognition settings
        self.confidence_threshold = config.slr_confidence_threshold
        self.frame_skip = 1
        
        # UI settings
        self.show_info_panel = True
        self.show_debug_info = False
        self.recognition_history = deque(maxlen=5)
        self.fps_history = deque(maxlen=30)
        
        # Current states
        self.current_result: Optional[RecognitionResult] = None
        self.latest_recognition_text = ""
        self.queue_sizes = {'frame': 0, 'slr': 0, 'tts': 0}
        
        # Performance monitoring
        self.performance_stats = {
            'frames_processed': 0,
            'recognitions_made': 0,
            'tts_triggered': 0,
            'start_time': time.time()
        }
        
    def initialize_camera(self, camera_index=0):
        """Initialize webcam with optimized settings"""
        self.cap = cv2.VideoCapture(camera_index)
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Target 30 FPS
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frames
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_index}")
        
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera initialized: {int(actual_width)}x{int(actual_height)} @ {actual_fps:.1f} FPS")
        
    def start_workers(self):
        """Start all worker threads"""
        # Start Cam worker
        self.cam_worker = CameraWorker(self.video_queue, self.stop_event, self.cap)
        self.cam_worker.initialize()
        self.cam_worker.start()
        
        # Start SLR worker
        self.slr_worker = SLRWorker(self.video_queue, self.slr_queue, self.stop_event, device='cpu')
        self.slr_worker.initialize()
        self.slr_worker.start()
        
        # Start TTS worker
        self.tts_worker = TTSWorker(self.tts_queue)
        self.slr_worker.initialize()
        self.tts_worker.start()
        
        print("All worker threads started")
        
    def stop_workers(self):
        """Stop all worker threads gracefully"""
        # Stop workers
        if self.slr_worker:
            self.slr_worker.close()
            self.slr_worker.join(timeout=1.0)
            
        if self.tts_worker:
            self.tts_worker.close()
            self.tts_worker.join(timeout=1.0)
            
        if self.cam_worker:
            self.cam_worker.close()
            self.cam_worker.join(timeout=1.0)
        
        # Send poison pills
        self.video_queue.put(None)
        self.tts_queue.put(None)
        
            
        print("All worker threads stopped")
    
    
    def check_recognition_results(self):
        """Check for new recognition results (non-blocking)"""
        try:
            while not self.slr_queue.empty():
                result = self.slr_queue.get_nowait()
                self.current_result = result
                
                if result['detected'] and result['confidence'] >= self.confidence_threshold:
                    # Add to history
                    self.recognition_history.append({
                        'text': result['text'],
                        'confidence': result['confidence'],
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'metadata': result['metadata']
                    })
                    
                    # Send to TTS queue
                    try:
                        self.tts_queue.put_nowait(result['text'])
                        self.performance_stats['tts_triggered'] += 1
                    except queue.Full:
                        pass
                    
                    self.latest_recognition_text = result['text']
                    self.performance_stats['recognitions_made'] += 1
                    
        except queue.Empty:
            pass
    
    def update_queue_sizes(self):
        """Update queue size monitoring"""
        self.queue_sizes = {
            'frame': self.video_queue.qsize(),
            'slr': self.slr_queue.qsize(),
            'tts': self.tts_queue.qsize()
        }
    
    def draw_ui(self, frame):
        """Draw UI elements on frame"""
        height, width = frame.shape[:2]
        
        # Draw main frame border
        cv2.rectangle(frame, (5, 5), (width-5, height-5), (0, 255, 0), 2)
        
        # Draw title
        cv2.putText(frame, "CSLR Demo - Real-time Sign Language Recognition", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw current recognition result
        if self.current_result and self.current_result['detected']:
            text = self.current_result['text']
            confidence = self.current_result['confidence']
            color = (0, 255, 0) if confidence >= self.confidence_threshold else (0, 165, 255)
            
            # Draw recognition box
            cv2.rectangle(frame, (10, 50), (width-10, 100), color, 2)
            cv2.putText(frame, f"Detected: {text}", 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Confidence: {confidence:.2%}", 
                       (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw info panel
        if self.show_info_panel:
            self._draw_info_panel(frame)
        
        # Draw debug info
        if self.show_debug_info:
            self._draw_debug_info(frame)
        
        # Draw FPS
        if len(self.fps_history) > 0:
            avg_fps = np.mean(self.fps_history)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", 
                       (width-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw controls help
        self._draw_controls(frame)
        
        return frame
    
    def _draw_info_panel(self, frame):
        """Draw information panel with recognition history"""
        height, width = frame.shape[:2]
        panel_width = 250
        panel_height = 200
        panel_x = width - panel_width - 10
        panel_y = height - panel_height - 80
        
        # Draw panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw panel border
        cv2.rectangle(frame, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (0, 255, 0), 1)
        
        # Panel title
        cv2.putText(frame, "Recognition History", 
                   (panel_x + 10, panel_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Draw history
        y_offset = 45
        for item in reversed(list(self.recognition_history)):
            cv2.putText(frame, f"{item['timestamp']}: {item['text']}", 
                       (panel_x + 10, panel_y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"  Conf: {item['confidence']:.2%}", 
                       (panel_x + 10, panel_y + y_offset + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
            y_offset += 35
    
    def _draw_debug_info(self, frame):
        """Draw debug information panel"""
        height, width = frame.shape[:2]
        debug_y = 120
        
        # Background for debug info
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, debug_y), (300, debug_y + 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Debug info
        runtime = time.time() - self.performance_stats['start_time']
        cv2.putText(frame, "=== Debug Info ===", (15, debug_y + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"Queues - F:{self.queue_sizes['frame']} R:{self.queue_sizes['slr']} T:{self.queue_sizes['tts']}", 
                   (15, debug_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Frames sent: {self.performance_stats['frames_processed']}", 
                   (15, debug_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Recognitions: {self.performance_stats['recognitions_made']}", 
                   (15, debug_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Runtime: {runtime:.1f}s", 
                   (15, debug_y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _draw_controls(self, frame):
        """Draw control instructions"""
        height, width = frame.shape[:2]
        controls = [
            "Q/ESC: Quit | I: Info Panel | D: Debug",
            "Space: Pause | R: Reset | +/-: Frame Skip"
        ]
        
        y_start = height - 30
        for i, control in enumerate(controls):
            cv2.putText(frame, control, 
                       (10, y_start + i*15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def run(self):
        """Main application loop - focuses on video capture and display"""
        print("Starting CSLR Demo...")
        print("Press Q or ESC to quit")
        
        try:
            # Initialize camera
            self.initialize_camera()
            
            # Start worker threads
            self.start_workers()
            
            # Main loop variables
            paused = False
            prev_time = time.time()
            frame = None
            
            print("\n=== MAIN LOOP STARTED ===")
            print("Video capture running in main thread")
            print("SLR and TTS running in separate threads\n")
            
            while True:
                # ALWAYS capture frames (unless paused)
                if not paused:
                    ret, new_frame = self.cap.read()
                    if not ret:
                        print("Failed to grab frame")
                        break
                    
                    # Flip frame horizontally for mirror effect
                    frame = cv2.flip(new_frame, 1)
                    
                    
                    # Check for recognition results (non-blocking)
                    self.check_recognition_results()
                    
                    # Update monitoring
                    self.update_queue_sizes()
                    
                    # Calculate FPS
                    current_time = time.time()
                    fps = 1 / (current_time - prev_time) if current_time != prev_time else 30
                    prev_time = current_time
                    self.fps_history.append(fps)
                    
                    # Draw UI
                    frame = self.draw_ui(frame)
                else:
                    # When paused, still display the last frame
                    if frame is not None:
                        display_frame = frame.copy()
                        cv2.putText(display_frame, "PAUSED", 
                                   (frame.shape[1]//2 - 50, frame.shape[0]//2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        frame = display_frame
                
                # Display frame (always happens)
                if frame is not None:
                    cv2.imshow('CSLR Demo - Real-time Sign Language Recognition', frame)
                
                # Handle keyboard input (non-blocking)
                key = cv2.waitKey(1) & 0xFF
                
                if key in [ord('q'), 27]:  # Q or ESC
                    print("\nQuitting...")
                    break
                elif key == ord('i'):  # Toggle info panel
                    self.show_info_panel = not self.show_info_panel
                    print(f"Info panel: {'ON' if self.show_info_panel else 'OFF'}")
                elif key == ord('d'):  # Toggle debug info
                    self.show_debug_info = not self.show_debug_info
                    print(f"Debug info: {'ON' if self.show_debug_info else 'OFF'}")
                elif key == ord(' '):  # Space - pause/resume
                    paused = not paused
                    print(f"{'PAUSED' if paused else 'RESUMED'}")
                elif key == ord('r'):  # Reset history
                    self.recognition_history.clear()
                    print("History cleared")
                elif key == ord('+') or key == ord('='):  # Increase frame skip
                    self.frame_skip = min(30, self.frame_skip + 5)
                    print(f"Frame skip: {self.frame_skip} (processing every {self.frame_skip}th frame)")
                elif key == ord('-'):  # Decrease frame skip
                    self.frame_skip = max(1, self.frame_skip - 5)
                    print(f"Frame skip: {self.frame_skip} (processing every {self.frame_skip}th frame)")
                    
        except Exception as e:
            print(f"Error in main loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        
        # Stop workers first
        self.stop_workers()
        
        # Release camera
        if self.cap:
            self.cap.release()
            
        # Destroy windows
        cv2.destroyAllWindows()
        
        # Print statistics
        runtime = time.time() - self.performance_stats['start_time']
        print("\n=== Performance Statistics ===")
        print(f"Total runtime: {runtime:.1f} seconds")
        print(f"Frames processed: {self.performance_stats['frames_processed']}")
        print(f"Recognitions made: {self.performance_stats['recognitions_made']}")
        print(f"TTS triggered: {self.performance_stats['tts_triggered']}")
        if runtime > 0:
            print(f"Avg processing rate: {self.performance_stats['frames_processed']/runtime:.1f} frames/sec")
        
        print("\nDemo ended successfully")