# Dataloader and image utils
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import IterableDataset
from ..configs import data_config
from ..utils import read_image

config = data_config


# load LAION art dataset
dataset_id = "ILSVRC/imagenet-1k"  # "laion/laion-art"  # "CortexLM/midjourney-v6"
hfdata = load_dataset(dataset_id, split="train", streaming=True, trust_remote_code=True)
hfdata = hfdata.take(100_000)


class ImageDataset(IterableDataset):
    def __init__(self, dataset: Dataset = hfdata):
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        for item in self.dataset:
            image = read_image(item["image"])
            class_label = item["label"]

            image = torch.tensor(image, dtype=config.dtype)
            class_label = torch.tensor(class_label, dtype=config.dtype)

            yield image, class_label
