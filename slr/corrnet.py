from slr.slr_network import SLRModel

from collections import OrderedDict
import numpy as np
import torch
from slr.config import DICT_PATH, MODEL_PATH, DEVICE_ID
from slr.utils import video_augmentation

class CorrModel():
    def __init__(self, device: str = 'cpu'):
        self.gloss_dict = np.load(DICT_PATH, allow_pickle=True).item()
        self.model = SLRModel(
            num_classes=len(self.gloss_dict)+1, c2d_type='resnet18', conv_type=2, use_bn=1, gloss_dict=self.gloss_dict,
            loss_weights={'ConvCTC': 1.0, 'SeqCTC': 1.0, 'Dist': 25.0}
        )

        state_dict = torch.load(MODEL_PATH, weights_only=False)['model_state_dict']
        state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
        self.model.load_state_dict(state_dict, strict=True)
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()

    def inference(self, video: list) -> dict:
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
            vid = vid.to(self.device)
            video_length = video_length.to(self.device)
            with torch.no_grad():
                ret_dict = self.model.forward(vid, video_length, label=None, label_lgt=None)
                return ret_dict
