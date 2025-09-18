"""
Simple Text-to-Speech Module using ElevenLabs API
"""
import numpy as np
import time
import logging
import threading
from typing import Optional, Callable
import io

import soundfile as sf
import sounddevice as sd
from elevenlabs import set_api_key, generate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_VOICE = "Rachel"
DEFAULT_MODEL = "eleven_multilingual_v1"
DEFAULT_SAMPLE_RATE = 22050

class ElevenLabsTTS:
    def __init__(self, api_key: str, voice: str = DEFAULT_VOICE, model: str = DEFAULT_MODEL):
        """
        Initialize ElevenLabs TTS
        
        Args:
            api_key: Your ElevenLabs API key - PUT YOUR API KEY HERE
            voice: Voice name to use
            model: Model name to use
        """
        self.voice = voice
        self.model = model
        
        # Set your API key here
        set_api_key(api_key)
        self.api_enabled = True
        logger.info("ElevenLabs TTS initialized successfully")
    
    def text_to_speech(self, txt: str) -> np.ndarray:
        """
        Convert text to speech using ElevenLabs API
        
        Args:
            txt: Input text to convert to speech
            
        Returns:
            np.ndarray: Audio data as float32 array
        """
        if not txt.strip():
            return np.array([])
        
        logger.info(f"TTS: Converting text to speech: '{txt[:50]}{'...' if len(txt) > 50 else ''}'")
        
        try:
            # Generate audio using ElevenLabs
            audio_data = generate(
                text=txt,
                voice=self.voice,
                model=self.model,
                stream=False
            )
            
            # Convert audio bytes to numpy array
            if isinstance(audio_data, bytes):
                # Use BytesIO to read the audio data
                with io.BytesIO(audio_data) as audio_buffer:
                    try:
                        audio_array, sample_rate = sf.read(audio_buffer)
                    except Exception:
                        # Fallback: treat as raw audio
                        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            else:
                # Handle generator case
                audio_bytes = b"".join(audio_data)
                with io.BytesIO(audio_bytes) as audio_buffer:
                    audio_array, sample_rate = sf.read(audio_buffer)
            
            # Convert to float32 if needed
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Handle stereo to mono conversion if needed
            if audio_array.ndim > 1:
                audio_array = np.mean(audio_array, axis=1)
            
            logger.info(f"TTS: Successfully generated audio ({len(audio_array)} samples)")
            return audio_array
            
        except Exception as e:
            logger.error(f"ElevenLabs TTS error: {e}")
            return self._generate_dummy_audio(txt)
    
    def _generate_dummy_audio(self, txt: str) -> np.ndarray:
        """Generate dummy audio data if API fails"""
        logger.info(f"[DUMMY TTS] Would speak: '{txt}'")
        time.sleep(0.1)  # Simulate processing time
        
        # Generate dummy audio (silence)
        duration = max(1.0, len(txt) * 0.08)
        samples = int(duration * DEFAULT_SAMPLE_RATE)
        return np.zeros(samples, dtype=np.float32)
    
    def text_to_speech_async(self, txt: str, callback: Optional[Callable] = None) -> threading.Thread:
        """
        Asynchronous TTS for non-blocking operation
        
        Args:
            txt: Text to convert
            callback: Optional callback function when TTS completes
        """
        def _tts_worker():
            try:
                audio_data = self.text_to_speech(txt)
                if callback:
                    callback(audio_data)
            except Exception as e:
                logger.error(f"Async TTS error: {e}")
                if callback:
                    callback(np.array([]))
        
        thread = threading.Thread(target=_tts_worker)
        thread.daemon = True
        thread.start()
        return thread
    
    def play_audio(self, audio_data: np.ndarray, sample_rate: int = DEFAULT_SAMPLE_RATE):
        """
        Play audio data using sounddevice
        
        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate of the audio
        """
        try:
            if len(audio_data) > 0:
                sd.play(audio_data, sample_rate)
                sd.wait()  # Wait until audio finishes playing
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
    
    def speak(self, txt: str):
        """
        Convert text to speech and play it immediately (simple method)
        
        Args:
            txt: Text to convert and play
        """
        audio_data = self.text_to_speech(txt)
        if len(audio_data) > 0:
            self.play_audio(audio_data)
    
    def set_voice(self, voice_name: str):
        """Set TTS voice"""
        self.voice = voice_name
        logger.info(f"Voice set to: {voice_name}")


if __name__ == "__main__":
    API_KEY = "2747bcf0e2a3b73607e6ca16828166d32fcc5e399b74f8c2faa88b8386ab2916" 
    
    tts = ElevenLabsTTS(api_key=API_KEY)
    
    tts.speak("Hello! This is a test of ElevenLabs text to speech.")
    
    audio = tts.text_to_speech("This is another test.")
    print(f"Generated audio with {len(audio)} samples")
    
    tts.play_audio(audio)
    
    tts.set_voice("Adam")  
    tts.speak("This is using Adam's voice.")


