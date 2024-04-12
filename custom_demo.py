import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

version="v1"
backbone="ViTL14"
model = torch.hub.load("lpiccinelli-eth/UniDepth", "UniDepth", version=version, backbone=backbone, pretrained=True, trust_repo=True, force_reload=False)


image_path="/usr/stud/kaa/data/munich/scene2_20240116_28s_1024_576/images/00002.jpg"

# Move to CUDA, if any
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load the RGB image and the normalization will be taken care of by the model
rgb = torch.from_numpy(np.array(Image.open(image_path))).permute(2, 0, 1) # C, H, W

intrinsics = np.array([[535.2, 0., 512], [0., 534.9, 288], [0., 0., 1.]], dtype=np.float32)
intrinsics = torch.from_numpy(intrinsics)

predictions = model.infer(rgb)
predictions = model.infer(rgb, intrinsics)

# Metric Depth Estimation
depth = predictions["depth"]

# Point Cloud in Camera Coordinate
xyz = predictions["points"]

# Intrinsics Prediction
# intrinsics = predictions["intrinsics"]

plt.imsave("debug_unidepth.png", depth.cpu().squeeze())
plt.imsave("debug_inv_unidepth.png", (1/depth).cpu().squeeze())

print(depth)