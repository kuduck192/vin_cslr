import threading
import queue
import time
from tts.tts_api import TTSModule

class TTSWorker(threading.Thread):
    def __init__(self, result_queue: queue.Queue, stop_event: threading.Event, api_name:str):
        super().__init__(daemon=True)
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.api_name = api_name
        self.client = None

    def initialize(self):
        '''Initialize TTS api'''
        self.client = TTSModule(api_name=self.api_name)

    def run(self):
        print("[TTS Worker] Started")
        while not self.stop_event.is_set():
            try:
                if not self.result_queue.empty():
                    recognition_result = self.result_queue.get()
                    text = recognition_result.text
                    self.client.to_speech(text)
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"[TTS Worker] Error: {e}")

    def close(self):
        print("[TTS Worker] Stopped")
