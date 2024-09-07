# Dataloader and image utils
import torch
from skimage import io
import numpy as np
import cv2
import random
from datasets import Dataset, load_dataset
from torch.utils.data import IterableDataset
from ..configs import data_config

config = data_config


def read_image(img_url, img_size=512):
    img = io.imread(img_url) if isinstance(str, img_url) else np.array(img_url)
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0

    return img


# load LAION art dataset
dataset_id = "laion/laion-art"  # "CortexLM/midjourney-v6"
hfdata = load_dataset(dataset_id, split="train", streaming=True)
hfdata = hfdata.take(100_000)


class ImageDataset(IterableDataset):
    def __init__(self, dataset: Dataset = hfdata):
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        for item in self.dataset:
            image = read_image(item["image"])
            class_label = item["TEXT"]

            image = torch.tensor(image, dtype=config.dtype)
            class_label = torch.tensor(class_label, dtype=config.dtype)

            yield image, class_label


def seed_everything(seed=333):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
