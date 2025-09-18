import json
import os
import io
import sys
import soundfile as sf
import numpy as np
import sounddevice as sd
from elevenlabs import set_api_key, generate

# === CẤU HÌNH ===
VOICE = "Rachel"
MODEL = "eleven_multilingual_v1"

# === SET API KEY ===
set_api_key("2747bcf0e2a3b73607e6ca16828166d32fcc5e399b74f8c2faa88b8386ab2916") 

def read_input_json(json_path: str) -> str:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("text", "")

def synthesize_to_wav(text: str, wav_path: str) -> int:
    print(f"Synthesizing: {text}")
    try:
        audio_bytes = generate(
            text=text,
            voice=VOICE,
            model=MODEL,
            stream=False
        )
    except Exception as e:
        print(f"Lỗi khi gọi ElevenLabs API: {e}")
        sys.exit(1)

    buffer = io.BytesIO(audio_bytes)
    audio_data, sample_rate = sf.read(buffer, dtype="float32")
    sf.write(wav_path, audio_data, sample_rate)
    print(f"Saved to: {wav_path}")
    return sample_rate

def play_wav(wav_path: str):
    try:
        data, samplerate = sf.read(wav_path, dtype="float32")
        print(f"Playing audio ({len(data)/samplerate:.2f}s)...")
        sd.play(data, samplerate)
        sd.wait()
        print("Playback finished.")
    except Exception as e:
        print(f"Lỗi khi phát âm thanh: {e}")

def main():
    if len(sys.argv) < 4 or sys.argv[2] != "--out":
        print("Thiếu tham số.\n Cách dùng:")
        print("python tts_api.py <input.json> --out <output.wav>")
        sys.exit(1)

    input_json = sys.argv[1]
    output_wav = sys.argv[3]

    if not os.path.exists(input_json):
        print(f"Không tìm thấy file: {input_json}")
        sys.exit(1)

    text = read_input_json(input_json)
    if not text.strip():
        print("Văn bản trống.")
        sys.exit(1)

    synthesize_to_wav(text, output_wav)
    play_wav(output_wav)

if __name__ == "__main__":
    main()
