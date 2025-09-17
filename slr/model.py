from slr.corrnet import CorrModel

def init_model(model_name: str, device: str):
    if model_name == 'corrnet':
        return CorrModel(device=device)

