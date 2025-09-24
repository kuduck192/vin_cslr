#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, sys
import numpy as np
import sounddevice as sd
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from tts.config import API_KEY, VOICE_ID, MODEL_ID, SR

class TTSModule():
    def __init__(self, api_name: str):
        self.api_name = api_name
        self.client = self._init_api()

    def _init_api(self):
        if self.api_name == 'ElevenLab':
            return ElevenLabs(api_key=API_KEY)
        else:
            return None
    
    def to_speech(self, text: str):
        audio_stream = self.client.text_to_speech.convert(
            text=text,
            voice_id=VOICE_ID,
            model_id=MODEL_ID,
            output_format='pcm_22050'
        )
        with sd.OutputStream(samplerate=SR, channels=1, dtype="int16") as stream:
            for chunk in audio_stream:
                if not chunk:
                    continue
                frames = np.frombuffer(chunk, dtype=np.int16)
                stream.write(frames)

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data.get("text"), str):
        return data["text"].strip()
    if isinstance(data.get("lines"), list):
        return "\n".join(map(str, data["lines"])).strip()
    return ""

def main():
    if len(sys.argv) < 2:
        print("Usage: python tts_play.py <input.json>")
        sys.exit(1)

    text = read_text(sys.argv[1])
    if not text:
        print("Input JSON must contain 'text' (str) or 'lines' (list).")
        sys.exit(1)

    client = ElevenLabs(api_key=API_KEY)

    audio_stream = client.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id=MODEL_ID,
        output_format="pcm_22050",  
    )

    with sd.OutputStream(samplerate=SR, channels=1, dtype="int16") as stream:
        for chunk in audio_stream:
            if not chunk:
                continue
            frames = np.frombuffer(chunk, dtype=np.int16)
            stream.write(frames)

if __name__ == "__main__":
    main()
    print("Done.")
