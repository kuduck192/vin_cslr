import threading, queue, time
from tts import text_to_speech_async


class TTSWorker(threading.Thread):
    """Dedicated worker thread for Text-to-Speech"""
    
    def __init__(self, tts_queue: queue.Queue):
        super().__init__(daemon=True)
        self.tts_queue = tts_queue
        self.running = True
        self.last_spoken_text = ""
        self.last_speak_time = 0
        self.speak_cooldown = 2.0
    
    def initialize(self): pass
    
    def run(self):
        """Main TTS worker loop"""
        print("[TTS Worker] Started")
        
        while self.running:
            try:
                # Get text from queue
                text = self.tts_queue.get(timeout=0.1)
                
                if text is None:  # Poison pill
                    break
                
                # Check cooldown to avoid repeating
                current_time = time.time()
                should_speak = (
                    text != self.last_spoken_text or
                    current_time - self.last_speak_time > self.speak_cooldown
                )
                
                if should_speak:
                    # Perform TTS (already async in the module)
                    text_to_speech_async(text)
                    self.last_spoken_text = text
                    self.last_speak_time = current_time
                    print(f"[TTS] Speaking: '{text}'")
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[TTS Worker] Error: {e}")
        
        print("[TTS Worker] Stopped")
    
    def close(self):
        """Stop the worker thread"""
        self.running = False