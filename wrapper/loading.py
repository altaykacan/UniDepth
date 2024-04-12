from pathlib import Path

import torch

def get_detpth_config(cfg_path: Path = None, weigth_path: Path = None):
    """
    Returns a dummy config object that is only used to fit the template
    # TODO change and make classes for this!
    """
    cfg = ""

    return cfg

def get_depth_model(cfg):
    """
    Instantiate an UniDepth model from a dummy config object.
    """
    version="v1"
    backbone="ViTL14"

    model = torch.hub.load("lpiccinelli-eth/UniDepth", "UniDepth", version=version, backbone=backbone, pretrained=True, trust_repo=True, force_reload=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()
