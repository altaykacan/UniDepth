from typing import Union

import torch
import numpy as np
from PIL import Image

def predict_depth(model, cfg, image: Union[Image.Image, np.ndarray], intrinsics: tuple, predict_normals=False):
    assert predict_normals == False, "Can't predict normals using UniDepth! Please set predict_normals to false."

    if isinstance(image, Image.Image):
        image = np.asarray(image)

    fx, fy, cx, cy = intrinsics
    intrinsics = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.0]])


    preds = model.infer(image, intrinsics)
    depth = preds["depth"]

    return depth



