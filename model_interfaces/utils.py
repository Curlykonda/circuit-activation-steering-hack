import torch

_DEVICE = None


def _set_device():
    global _DEVICE
    _DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def get_device() -> torch.device:
    if _DEVICE is None:
        _set_device()
    return _DEVICE
