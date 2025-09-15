"""
Real-time Continuous Sign Language Recognition (CSLR) Demo
Video -> SLR -> TTS -> Audio Pipeline
"""
import cv2
import numpy as np
import time
from collections import deque
import threading
from datetime import datetime

# Import custom modules
from slr import sign_language_recognition
from tts import text_to_speech_async


class CSLRDemo:
    def __init__(self):
        """Initialize CSLR demo application"""
        # Video capture
        self.cap = None
        self.frame_width = 640
        self.frame_height = 480
        
        # Recognition settings
        self.recognition_interval = 0.5  # Process every 0.5 seconds
        self.last_recognition_time = 0
        self.last_spoken_text = ""
        self.speak_cooldown = 2.0  # Avoid repeating same phrase within 2 seconds
        self.last_speak_time = 0
        
        # UI settings
        self.show_info_panel = True
        self.recognition_history = deque(maxlen=5)
        self.fps_history = deque(maxlen=30)
        
        # Recognition results
        self.current_result = None
        self.confidence_threshold = 0.6
        
        # Thread safety
        self.lock = threading.Lock()
        
    def initialize_camera(self, camera_index=0):
        """Initialize webcam"""
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_index}")
        
        print(f"Camera initialized: {self.frame_width}x{self.frame_height}")
        
    def process_frame(self, frame):
        """Process frame for sign language recognition"""
        current_time = time.time()
        
        # Check if it's time to run recognition
        if current_time - self.last_recognition_time >= self.recognition_interval:
            # Run SLR in a separate thread to avoid blocking
            thread = threading.Thread(target=self._recognize_async, args=(frame.copy(),))
            thread.daemon = True
            thread.start()
            self.last_recognition_time = current_time
    
    def _recognize_async(self, frame):
        """Asynchronous recognition processing"""
        result = sign_language_recognition(frame)
        
        with self.lock:
            self.current_result = result
            
            # If valid recognition with high confidence
            if result['detected'] and result['confidence'] >= self.confidence_threshold:
                recognized_text = result['text']
                
                # Add to history
                self.recognition_history.append({
                    'text': recognized_text,
                    'confidence': result['confidence'],
                    'timestamp': datetime.now().strftime("%H:%M:%S")
                })
                
                # Check if we should speak this text
                current_time = time.time()
                should_speak = (
                    recognized_text != self.last_spoken_text or
                    current_time - self.last_speak_time > self.speak_cooldown
                )
                
                if should_speak:
                    # Trigger TTS
                    text_to_speech_async(recognized_text)
                    self.last_spoken_text = recognized_text
                    self.last_speak_time = current_time
    
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
        panel_y = height - panel_height - 60
        
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
    
    def _draw_controls(self, frame):
        """Draw control instructions"""
        height, width = frame.shape[:2]
        controls = [
            "Q/ESC: Quit",
            "I: Toggle Info Panel",
            "Space: Pause/Resume",
            "R: Reset History"
        ]
        
        y_start = height - 50
        for i, control in enumerate(controls):
            cv2.putText(frame, control, 
                       (10, y_start + i*15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def run(self):
        """Main application loop"""
        print("Starting CSLR Demo...")
        print("Press Q or ESC to quit")
        
        try:
            self.initialize_camera()
            paused = False
            prev_time = time.time()
            
            while True:
                if not paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Failed to grab frame")
                        break
                    
                    # Flip frame horizontally for mirror effect
                    frame = cv2.flip(frame, 1)
                    
                    # Process frame for recognition
                    self.process_frame(frame)
                    
                    # Calculate FPS
                    current_time = time.time()
                    fps = 1 / (current_time - prev_time)
                    prev_time = current_time
                    self.fps_history.append(fps)
                    
                    # Draw UI
                    frame = self.draw_ui(frame)
                else:
                    # Show paused message
                    cv2.putText(frame, "PAUSED", 
                               (frame.shape[1]//2 - 50, frame.shape[0]//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Display frame
                cv2.imshow('CSLR Demo - Real-time Sign Language Recognition', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key in [ord('q'), 27]:  # Q or ESC
                    break
                elif key == ord('i'):  # Toggle info panel
                    self.show_info_panel = not self.show_info_panel
                elif key == ord(' '):  # Space - pause/resume
                    paused = not paused
                elif key == ord('r'):  # Reset history
                    self.recognition_history.clear()
                    
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Demo ended")


def main():
    """Entry point"""
    print("="*50)
    print("CSLR Demo - Continuous Sign Language Recognition")
    print("="*50)
    print("\nInitializing system...")
    
    # Create and run demo
    demo = CSLRDemo()
    demo.run()


if __name__ == "__main__":
    main()