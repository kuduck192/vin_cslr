"""
Text-to-Speech Module (iFlyTek)
Synchronous + Asynchronous TTS with WAV output & playback
"""

import json, base64, hmac, hashlib, ssl, logging, wave, threading, sys
from urllib.parse import urlencode
from datetime import datetime
from time import mktime
from wsgiref.handlers import format_date_time

import websocket
from playsound import playsound

# ===== Logging =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === API Config (iFlyTek) ===
APPID = "ga7aa423"
APIKey = "f1a47b56ed5bd951ee1165fbfa13b3ef"
APISecret = "bb110a7043d9587e0906abcabd6be8b7"

class Ws_Param:
    def __init__(self, APPID, APIKey, APISecret, Text):
        self.CommonArgs = {"app_id": APPID}
        self.BusinessArgs = {
            "aue": "raw", "auf": "audio/L16;rate=16000",
            "vcn": "x_John", "tte": "utf8"
        }
        self.Data = {"status": 2, "text": base64.b64encode(Text.encode()).decode()}
        self.APIKey, self.APISecret = APIKey, APISecret

    def create_url(self):
        host = "tts-api-sg.xf-yun.com"
        path = "/v2/tts"
        url = f"wss://{host}{path}"

        date = format_date_time(mktime(datetime.now().timetuple()))
        sig_origin = f"host: {host}\ndate: {date}\nGET {path} HTTP/1.1"
        sig_sha = hmac.new(self.APISecret.encode(), sig_origin.encode(), hashlib.sha256).digest()
        signature = base64.b64encode(sig_sha).decode()

        auth_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature}"'
        authorization = base64.b64encode(auth_origin.encode()).decode()

        v = {"authorization": authorization, "date": date, "host": host}
        return url + "?" + urlencode(v)


def _call_iflytek_api(txt: str) -> bytes:
    wsParam = Ws_Param(APPID, APIKey, APISecret, txt)
    wsUrl = wsParam.create_url()
    audio_chunks = []

    def on_message(ws, message):
        data = json.loads(message)
        if data["code"] != 0:
            logger.error(f"TTS error: {data['message']} (code {data['code']})")
            ws.close(); return
        audio_chunks.append(base64.b64decode(data["data"]["audio"]))
        if data["data"]["status"] == 2:
            ws.close()

    ws = websocket.WebSocketApp(
        wsUrl, on_message=on_message,
        on_error=lambda ws, err: logger.error(f"WebSocket error: {err}"),
        on_close=lambda *a: logger.info("WebSocket closed")
    )
    ws.on_open = lambda ws: ws.send(json.dumps({
        "common": wsParam.CommonArgs,
        "business": wsParam.BusinessArgs,
        "data": wsParam.Data
    }))
    ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})

    return b"".join(audio_chunks)


def text_to_speech(txt: str, out_path="output.wav", play=True) -> str:
    """Sync TTS: save WAV + optional play. Returns output file path."""
    if not txt:
        return ""
    try:
        logger.info(f"[TTS] Generating audio for: {txt}")
        audio_bytes = _call_iflytek_api(txt)

        with wave.open(out_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_bytes)

        logger.info(f"[TTS] Saved {out_path}")
        if play:
            playsound(out_path)
        return out_path
    except Exception as e:
        logger.error(f"TTS error: {e}")
        return ""


def text_to_speech_async(txt: str, callback=None, out_path="output.wav", play=True):
    """Async TTS: run in thread, optional callback with out_path."""
    def _worker():
        path = text_to_speech(txt, out_path=out_path, play=play)
        if callback:
            try:
                callback(path)
            except Exception as e:
                logger.error(f"TTS callback error: {e}")

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data.get("text"), str):
        return data["text"].strip()
    if isinstance(data.get("lines"), list):
        return "\n".join(map(str, data["lines"])).strip()
    return ""


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "tts/input.json"
    text = read_text(path)
    if not text:
        print("Input JSON phải có 'text' (str) hoặc 'lines' (list)."); sys.exit(1)
    text_to_speech(text)
