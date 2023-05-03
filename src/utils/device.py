import torch

def get_device():
    """Get type of device being used for training either cpu or cuda."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device