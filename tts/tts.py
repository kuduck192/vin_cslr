"""
Text-to-Speech Module
ElevenLabs realtime playback (no file) with graceful fallbacks.
Keeps the original API shape of the dummy module.
"""
import time
import json
import sys
import logging
import threading
from typing import Optional, List

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SD_AVAILABLE = False
try:
    import sounddevice as sd
    SD_AVAILABLE = True
except Exception:
    pass

TTS_AVAILABLE = False
engine = None
try:
    import pyttsx3  
    engine = pyttsx3.init()
    TTS_AVAILABLE = True
except Exception:
    engine = None
    TTS_AVAILABLE = False

# ElevenLabs (cloud TTS primary)
EL_AVAILABLE = False
EL_CLIENT = None
try:
    from elevenlabs.client import ElevenLabs
    EL_AVAILABLE = True
except Exception:
    EL_AVAILABLE = False

API_KEY  = "2747bcf0e2a3b73607e6ca16828166d32fcc5e399b74f8c2faa88b8386ab2916"   # <-- Thay bằng key thật
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"               # Rachel
MODEL_ID = "eleven_multilingual_v2"
SR       = 22050                                # PCM sample rate (must match output_format)
CHANNELS = 1                                    # mono
SAMPWIDTH = 2                                   # 16-bit PCM

def _init_elevenlabs_client():
    """Initialize ElevenLabs client once."""
    global EL_CLIENT
    if not EL_AVAILABLE:
        return None
    if EL_CLIENT is None:
        if not API_KEY or API_KEY.startswith("PUT_"):
            logger.warning("ElevenLabs API key is not set properly; fallback will be used.")
            return None
        try:
            EL_CLIENT = ElevenLabs(api_key=API_KEY)
        except Exception as e:
            logger.error(f"Failed to init ElevenLabs client: {e}")
            EL_CLIENT = None
    return EL_CLIENT


def _eleven_stream_pcm(text: str):
    """
    Yield PCM 16-bit little-endian bytes chunks from ElevenLabs.
    """
    client = _init_elevenlabs_client()
    if client is None:
        return None
    try:
        return client.text_to_speech.convert(
            text=text,
            voice_id=VOICE_ID,
            model_id=MODEL_ID,
            output_format="pcm_22050",  # matches SR
        )
    except Exception as e:
        logger.error(f"ElevenLabs convert() error: {e}")
        return None


def _float32_from_int16(int16_array: np.ndarray) -> np.ndarray:
    """Convert PCM int16 to float32 (-1..1)."""
    if int16_array.size == 0:
        return np.zeros(0, dtype=np.float32)
    return (int16_array.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def text_to_speech(txt: str) -> np.ndarray:
    """
    Convert text to speech and (in this implementation) PLAY IT LIVE.
    Returns the full audio as float32 (-1..1) after playback completes.

    Fallback order:
    1) ElevenLabs -> stream PCM via sounddevice
    2) pyttsx3 local engine (no audio array, returns silence of estimated duration)
    3) Dummy: log & return silence
    """
    if not txt:
        return np.zeros(0, dtype=np.float32)

    logger.info(f"TTS: Converting text '{txt}' to speech")

    if EL_AVAILABLE and SD_AVAILABLE:
        stream = _eleven_stream_pcm(txt)
        if stream is not None:
            logger.info("Using ElevenLabs (PCM 22.05kHz) + sounddevice for live playback")
            collected: List[np.ndarray] = []
            try:
                with sd.OutputStream(samplerate=SR, channels=CHANNELS, dtype="int16") as out:
                    for chunk in stream:
                        if not chunk:
                            continue
                        frames = np.frombuffer(chunk, dtype=np.int16)
                        out.write(frames)
                        collected.append(frames)
                if collected:
                    all_int16 = np.concatenate(collected)
                    return _float32_from_int16(all_int16)
                else:
                    return np.zeros(0, dtype=np.float32)
            except Exception as e:
                logger.error(f"Playback error (sounddevice/stream): {e}")

    if TTS_AVAILABLE and engine:
        logger.info("Using local pyttsx3 fallback")
        try:
            engine.say(txt)
            engine.runAndWait()
        except Exception as e:
            logger.error(f"pyttsx3 error: {e}")
        duration = max(0.5, len(txt) * 0.06)  # rough estimate ~60ms/char
        samples = int(duration * 16000)
        return np.zeros(samples, dtype=np.float32)

    logger.info(f"[DUMMY TTS] Would speak: '{txt}'")
    time.sleep(min(3.0, max(0.1, len(txt) * 0.02)))
    duration = max(0.5, len(txt) * 0.06)
    samples = int(duration * 16000)
    return np.zeros(samples, dtype=np.float32)


def text_to_speech_async(txt: str, callback=None):
    """
    Asynchronous TTS for non-blocking operation.
    Plays audio during the process (if ElevenLabs+sounddevice available).
    The callback (if provided) receives the float32 waveform.
    """
    def _tts_worker():
        audio = text_to_speech(txt)
        if callback:
            try:
                callback(audio)
            except Exception as e:
                logger.error(f"TTS callback error: {e}")

    thread = threading.Thread(target=_tts_worker, daemon=True)
    thread.start()
    return thread


def set_voice(voice_id: str):
    """
    Set TTS voice.
    - ElevenLabs: accepts a full voice_id string, e.g. '21m00Tcm4TlvDq8ikWAM'
    - pyttsx3: if given an integer index (string digits), will attempt to select by index; else no-op.
    """
    global VOICE_ID
    if EL_AVAILABLE:
        VOICE_ID = str(voice_id)
        logger.info(f"ElevenLabs VOICE_ID set to: {VOICE_ID}")
        return

    if TTS_AVAILABLE and engine:
        try:
            if voice_id.isdigit():
                idx = int(voice_id)
                voices = engine.getProperty('voices')
                if 0 <= idx < len(voices):
                    engine.setProperty('voice', voices[idx].id)
                    logger.info(f"pyttsx3 voice set to index {idx}")
        except Exception as e:
            logger.error(f"pyttsx3 set_voice error: {e}")


def set_rate(rate: int):
    """Set speech rate (pyttsx3 only). Not applicable to ElevenLabs via this simple wrapper."""
    if TTS_AVAILABLE and engine:
        try:
            engine.setProperty('rate', int(rate))
        except Exception as e:
            logger.error(f"pyttsx3 set_rate error: {e}")


def set_volume(volume: float):
    """Set volume (0.0 to 1.0) for pyttsx3 only."""
    if TTS_AVAILABLE and engine:
        try:
            engine.setProperty('volume', float(max(0.0, min(1.0, volume))))
        except Exception as e:
            logger.error(f"pyttsx3 set_volume error: {e}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "tts/input.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    text = (data.get("text") or "\n".join(map(str, data.get("lines", [])))).strip()
    if not text:
        print("Input JSON phải có 'text' (str) hoặc 'lines' (list)."); sys.exit(1)
    audio = text_to_speech(text) 
    print(f"Done. Returned samples: {audio.shape}, dtype={audio.dtype}")