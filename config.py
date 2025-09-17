class Config:

    # cv2
    frame_width = 800
    frame_height = 600
    
    # 3 threads: UI, SLR, TTS
    frame_queue_size = 200
    slr_queue_size = 50
    tts_queue_size = 50
    
    # Cam/UI
    
    # SLR
    slr_confidence_threshold = 0.5
    video_length_sec = 2
    min_frame_to_infer = 4
    


config = Config()