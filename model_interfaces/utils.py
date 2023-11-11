import torch

_DEVICE = None


def _set_device():
    global _DEVICE
    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_device():
    if _DEVICE is None:
        _set_device()
    return _DEVICE
