import torch
from PIL import Image
from torch import nn
from torchvision.transforms import ToTensor


class AdaptedImgToTensor(nn.Module):
    """
    Adapts the data pipeline for mixed grayscale and rgb images. Just return the Tensor of PIL image
    if image is RGB, otherwise duplicate to three channels grayscale image and return the tensor.
    """

    def forward(self, img: Image) -> torch.Tensor:
        if img.mode == "L":
            img_tensor = ToTensor()(img)
            img_tensor_3ch = img_tensor.expand(3, -1, -1)
            return img_tensor_3ch
        return ToTensor()(img)
