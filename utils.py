import torch
import random
from typing import Union
from PIL.Image import Image
from skimage import io
import cv2
import numpy as np


def modulate(x, shift_value, scale_value):
    x = x * (1 + scale_value.unsqueeze(1))
    x = x + shift_value.unsqueeze(1)

    return x


def read_image(img_url: Union[str, Image], img_size: int = 512):
    img = io.imread(img_url) if isinstance(str, img_url) else np.array(img_url)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0

    return img


def seed_everything(seed=333):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
