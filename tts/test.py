
import logging, threading, json, base64, hmac, hashlib, ssl, os
from urllib.parse import urlencode
from datetime import datetime
from time import mktime
from wsgiref.handlers import format_date_time

import websocket              # from websocket-client
from playsound import playsound      
import wave                   # xử lý file WAV

# ===== Cấu hình logging =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === API Config (iFlyTek) ===
APPID = "ga7aa423"
APIKey = "f1a47b56ed5bd951ee1165fbfa13b3ef"
APISecret = "bb110a7043d9587e0906abcabd6be8b7"

# ===== Create URL to connect to WebSocket =====
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

        # Date and time RFC1123
        date = format_date_time(mktime(datetime.now().timetuple()))

        # signature
        sig_origin = f"host: {host}\ndate: {date}\nGET {path} HTTP/1.1"
        sig_sha = hmac.new(self.APISecret.encode(), sig_origin.encode(), hashlib.sha256).digest()
        signature = base64.b64encode(sig_sha).decode()

        # authorization
        auth_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature}"'
        authorization = base64.b64encode(auth_origin.encode()).decode()

        # query params
        v = {"authorization": authorization, "date": date, "host": host}
        return url + "?" + urlencode(v)



def _call_iflytek_api(txt: str) -> bytes:
    """
    Call iFlyTek TTS API and return audio bytes (PCM)
    Args:
        txt: Input text to convert to speech
    Returns:
        bytes: Audio data in PCM format
    """

    wsParam = Ws_Param(APPID, APIKey, APISecret, txt)
    wsUrl = wsParam.create_url()
    audio_chunks = []

    def on_message(ws, message):
        data = json.loads(message)
        if data["code"] != 0:
            logger.error(f"TTS error: {data['message']} (code {data['code']})")
            ws.close(); return
        audio_chunks.append(base64.b64decode(data["data"]["audio"]))
        if data["data"]["status"] == 2:  # kết thúc
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


def text_to_speech(txt: str):
    """
    Convert text to speech/audio
    Args:
        txt: Input text to convert to speech 
    Returns:
        Audio data
    """
    if not txt: return
    try:
        logger.info(f"[TTS] Generating audio for: {txt}")
        audio_bytes = _call_iflytek_api(txt)

        # Convert to wav and play
        with wave.open("output.wav", "wb") as wf:
            wf.setnchannels(1)        # mono
            wf.setsampwidth(2)        # 16-bit (2 bytes)
            wf.setframerate(16000)    # 16kHz
            wf.writeframes(audio_bytes)

        logger.info("[TTS] Saved output.wav")

        playsound("output.wav")
    except Exception as e:
        logger.error(f"TTS error: {e}")

def text_to_speech_async(txt: str, callback=None):
    """
    Asynchronous TTS for non-blocking operation

    Args:
        txt: Text to convert
        callback: Optional callback function when TTS completes
    """
    import threading
    
    def _tts_worker():
        audio = text_to_speech(txt)
        if callback:
            callback(audio)
    
    thread = threading.Thread(target=_tts_worker)
    thread.daemon = True
    thread.start()
    return thread

def main():
    sample_text = "Good morning! Have a nice day!"
    text_to_speech(sample_text)
    
if __name__ == "__main__":
    main()