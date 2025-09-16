"""
Sign Language Recognition Module
Dummy implementation for development
"""
import numpy as np
import time
import random
from .slr_network import SLRModel
from slr import utils

from slr.config import MODEL_PATH, DICT_PATH, DEVICE_ID

import torch
from collections import OrderedDict
from slr.utils import video_augmentation

class CorrModel():
    def __init__(self):
        self.gloss_dict = np.load(DICT_PATH, allow_pickle=True).item()
        self.model = SLRModel(
            num_classes=len(self.gloss_dict)+1, c2d_type='resnet18', conv_type=2, use_bn=1, gloss_dict=self.gloss_dict,
            loss_weights={'ConvCTC': 1.0, 'SeqCTC': 1.0, 'Dist': 25.0}
        )

        state_dict = torch.load(MODEL_PATH, weights_only=False)['model_state_dict']
        state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.cuda()
        self.model.eval()
        self.device = utils.GpuDataParallel()
        self.device.set_device(DEVICE_ID)

    def inference(self, video: list) -> dict:
        # while not self.stop_event.is_set():
        # if self.video_buffer:
            # frames = self.video_buffer.pop(0)
            frames = video
            # Preprocess
            transform = video_augmentation.Compose([
                video_augmentation.CenterCrop(224),
                video_augmentation.Resize(1.0),
                video_augmentation.ToTensor(),
            ])
            vid, label = transform(frames, None, None)
            vid = vid.float() / 127.5 - 1
            vid = vid.unsqueeze(0)
            max_len = vid.size(1)
            video_length = torch.LongTensor([max_len])
            vid = self.device.data_to_device(vid)
            video_length = self.device.data_to_device(video_length)
            with torch.no_grad():
                ret_dict = self.model.forward(vid, video_length, label=None, label_lgt=None)
                return ret_dict
        # else:
        #     time.sleep(0.1)


def inference_corrNet(video: list) -> dict:
    '''
    Inference Corrnet Model

    Args:
        video: list of video frame(s)
    Returns:
        dict: Recognition results with text, confidence, and metadata
    '''
    model = CorrModel()
    ret_dict = model.inference(video)
    return ret_dict


import torch
import numpy as np
from typing import Any, List, Union

model = None  # cache singleton

def _to_cpu_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return torch.as_tensor(x)

def _extract_text(pred: Any) -> str:
    """
    Chuẩn hoá nhiều format có thể trả về từ decoder:
    - ['hello world'] -> 'hello world'
    - [['hello','world']] -> 'hello world'
    - ['hello','world'] -> 'hello world'
    - [] hoặc None -> ''
    """
    if pred is None:
        return ""
    if isinstance(pred, str):
        return pred
    if isinstance(pred, (list, tuple)):
        if len(pred) == 0:
            return ""
        first = pred[0]
        # list of tokens trong batch
        if isinstance(first, (list, tuple)):
            return " ".join(map(str, first))
        # batch size = 1, decoder trả về 1 string
        if isinstance(first, str) and len(pred) == 1:
            return first
        # trường hợp khác: nối hết lại
        return " ".join(map(str, pred))
    # fallback
    return str(pred)

def _compute_confidence(ret_dict: dict) -> float:
    """
    Ước lượng độ tự tin bằng cách:
    - softmax over classes cho 'sequence_logits' (T, B, C)
    - lấy max theo lớp mỗi timestep -> (T, B)
    - lấy trung bình theo thời gian cho mẫu đầu tiên trong batch
    """
    try:
        logits = ret_dict.get("sequence_logits", None)
        if logits is None:
            return 0.0
        logits = _to_cpu_tensor(logits)  # (T, B, C)
        if logits.dim() != 3:
            return 0.0
        probs = torch.softmax(logits, dim=-1)  # (T, B, C)

        # Cắt theo feat_len nếu có
        lgt = ret_dict.get("feat_len", None)
        if lgt is not None:
            lgt = _to_cpu_tensor(lgt)
            L = int(lgt[0]) if lgt.numel() > 0 else probs.shape[0]
        else:
            L = probs.shape[0]

        L = max(1, min(L, probs.shape[0]))  # an toàn
        # Lấy mẫu đầu tiên trong batch
        max_t = probs[:L, 0, :].max(dim=-1).values  # (L,)
        return float(max_t.mean().item())
    except Exception:
        return 0.0

def sign_language_recognition(video: List[np.ndarray]) -> dict:
    '''
    Cache then infer Corrnet Model

    Args:
        video: list các frame (BGR) gần nhất (cửa sổ ~60 frames)
    Returns:
        dict: {'detected', 'text', 'confidence', 'metadata'}
    '''
    global model
    if model is None:
        model = CorrModel()  # giả định đã set .eval() & load weights trong CorrModel

    # CorrModel.inference(video) cần trả ret_dict cùng keys như SLRModel.forward
    ret_dict = model.inference(video)

    # Lấy text từ recognized_sents (ưu tiên), fallback conv_sents
    text = _extract_text(ret_dict.get("recognized_sents"))
    if not text.strip():
        text = _extract_text(ret_dict.get("conv_sents"))

    confidence = _compute_confidence(ret_dict)
    detected = bool(text.strip())

    # metadata gọn nhẹ, có thể thêm gì bạn muốn
    feat_len = ret_dict.get("feat_len", None)
    feat_len = int(_to_cpu_tensor(feat_len)[0]) if feat_len is not None else None

    results = {
        "detected": detected,
        "text": text,
        "confidence": confidence,     # 0.0 ~ 1.0; demo threshold của bạn là 0.6
        "metadata": {
            "feat_len": feat_len,
            "window_frames": len(video),
            "decoder": "beam",
            "conf_method": "avg_max_softmax_over_time",
        },
    }
    return results



    
def sign_language_recognition_dummy(video: np.ndarray = None) -> dict:
    """
    Dummy SLR function that simulates sign language recognition
    
    Args:
        video: numpy array of video frame(s)
        
    Returns:
        dict: Recognition results with text, confidence, and metadata
    """
    # Simulate processing time
    time.sleep(0.01)
    
    # Dummy responses to simulate recognition
    dummy_phrases = [
        "Hello",
        "Thank you",
        "How are you",
        "Good morning",
        "I love you",
        "Please",
        "Sorry",
        "Yes",
        "No",
        ""  # Empty for no detection
    ]
    
    # Randomly select a phrase (with some probability of no detection)
    if random.random() > 0.7:  # 30% chance of detection
        recognized_text = random.choice(dummy_phrases[:-1])
        confidence = random.uniform(0.6, 0.95)
    else:
        recognized_text = ""
        confidence = 0.0
    
    # Create result dictionary
    result = {
        'detected': len(recognized_text) > 0,
        'text': recognized_text,
        'confidence': confidence,
        'metadata': {
            'timestamp': time.time(),
            'processing_time': 0.01,
            'landmarks': None  # Placeholder for hand landmarks
        },
    }
    
    return result


if __name__ == "__main__":
    from decord import VideoReader, cpu
    # Path to your video
    video_path = 'D:/CTAI/CV/CorrNet/30_Khanh_400-700_13-14-15_0119___center_device01_signer30_center_ord1_569.mp4'
    
    # Read video frames using decord
    vr = VideoReader(video_path, ctx=cpu(0))
    frames = [frame.asnumpy() for frame in vr]  # List of numpy arrays, shape (H, W, 3)
    
    # Run inference
    result = inference_corrNet(frames)
    print("SLR Result:", result)