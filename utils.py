import torch
import random
from typing import Union
from PIL.Image import Image
from skimage import io
import cv2
import numpy as np
import gc


def modulate(x, shift_value, scale_value):
    x = x * (1 + scale_value.unsqueeze(1))
    x = x + shift_value.unsqueeze(1)

    return x


def count_params(model: torch.nn.Module):
    p_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return p_count


def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()


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


# diffusion generation preview
# latents = get_pred_original_sample(sched, *args)
# decoded = pipe.image_processor.postprocess(taesd_dec(latents.float()).mul_(2).sub_(1))[0]
# preview_images.append(decoded)
# preview_images[0].save("images/preview_images_1.gif", save_all=True, append_images=preview_images[1:], duration=100, loop=0)
# HTML("<img src=../images/preview_images_1.gif />")
