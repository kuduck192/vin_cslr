# ui.py
import cv2
import numpy as np
import time

def draw_ui(frame, demo):
    height, width = frame.shape[:2]

    # Border và tiêu đề (giữ nguyên)
    cv2.rectangle(frame, (5,5), (width-5,height-5), (0,255,0), 2)
    cv2.putText(frame, "CSLR Demo - Continuous Sign Language Recognition", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # FPS
    fps = demo.performance_stats.get('fps', 0)
    cv2.putText(frame, f"FPS: {fps:.1f}", (width-120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    # Paused
    if getattr(demo, 'paused', False):
        cv2.putText(frame, "PAUSED", (width//2-50, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Kết quả nhận dạng hiện tại
    if demo.current_result and demo.current_result.detected:
        text = demo.current_result.text
        confidence = demo.current_result.confidence
        # color = (0,255,0) if confidence >= demo.confidence_threshold else (0,165,255)
        color = (0,255,0)
        cv2.rectangle(frame, (10,50),(width-10,100), color, 2)
        cv2.putText(frame, f"Detected: {text}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.2%}", (20,95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Info panel và debug panel
    if demo.show_info_panel:
        _draw_info_panel(frame, demo)
    if demo.show_debug_info:
        _draw_debug_info(frame, demo)

    # Controls
    _draw_controls(frame)

    return frame



def _draw_info_panel(frame, demo):
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
    for item in reversed(list(demo.recognition_history)):
        cv2.putText(frame, f"{item['timestamp']}: {item['text']}", 
                    (panel_x + 10, panel_y + y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"  Conf: {item['confidence']:.2%}", 
                    (panel_x + 10, panel_y + y_offset + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
        y_offset += 35


def _draw_debug_info(frame, demo):
    """Draw debug information panel"""
    height, width = frame.shape[:2]
    debug_y = 120

    # Background for debug info
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, debug_y), (300, debug_y + 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Debug info
    runtime = time.time() - demo.performance_stats['start_time']
    cv2.putText(frame, "=== Debug Info ===", (15, debug_y + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    queue_frame = demo.queue_sizes.get('frame', 0)
    queue_result = demo.queue_sizes.get('result', 0)
    queue_tts = demo.queue_sizes.get('tts', 0)
    cv2.putText(frame, f"Queues - F:{queue_frame} R:{queue_result} T:{queue_tts}",
                (15, debug_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"Frames sent: {demo.performance_stats['frames_processed']}", 
                (15, debug_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"Recognitions: {demo.performance_stats['recognitions_made']}", 
                (15, debug_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"Runtime: {runtime:.1f}s", 
                (15, debug_y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def _draw_controls(frame):
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
    