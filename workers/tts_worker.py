import threading
import queue
import time

class TTSWorker(threading.Thread):
    def __init__(self, tts_queue: "queue.Queue[str]", speak_cooldown: float = 2.0):
        super().__init__(name="TTSWorker", daemon=True)
        self.tts_queue = tts_queue
        self.speak_cooldown = speak_cooldown
        self.running = True
        self.last_spoken_text = ""
        self.last_speak_time = 0.0
        self._speak_async = None
        self._speak_sync = None

    def initialize(self):
        try:
            from tts.tts import text_to_speech_async, text_to_speech
            self._speak_async = text_to_speech_async
            self._speak_sync = text_to_speech
        except Exception:
            try:
                from tts import text_to_speech_async, text_to_speech
                self._speak_async = text_to_speech_async
                self._speak_sync = text_to_speech
            except Exception:
                self._speak_async = None
                self._speak_sync = None

    def run(self):
        print("[TTS Worker] Started")
        self.initialize()
        while self.running:
            try:
                text = self.tts_queue.get(timeout=0.1)
                if text is None:
                    break
                if not isinstance(text, str):
                    continue
                text = text.strip()
                if not text:
                    continue

                now = time.time()
                if text == self.last_spoken_text and (now - self.last_speak_time) <= self.speak_cooldown:
                    continue

                if self._speak_async:
                    self._speak_async(text)
                elif self._speak_sync:
                    threading.Thread(target=self._speak_sync, args=(text,), daemon=True).start()

                self.last_spoken_text = text
                self.last_speak_time = now
                print(f"[TTS] Speaking: '{text}'")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[TTS Worker] Error: {e}")
        print("[TTS Worker] Stopped")

    def close(self):
        self.running = False
        try:
            self.tts_queue.put_nowait(None)
        except Exception:
            pass
