# Dataloader and image utils
import torch
from datasets import load_dataset
from torch.utils.data import IterableDataset, get_worker_info
from torch.distributed import get_rank, get_world_size
from torch.utils.data.distributed import DistributedSampler
from ..configs import data_config
from ..utils import read_image

config = data_config


# load LAION art dataset
split = 100_000
dataset_id = "ILSVRC/imagenet-1k"  # "laion/laion-art"  # "CortexLM/midjourney-v6"
hfdata = load_dataset(dataset_id, split="train", streaming=True, trust_remote_code=True)
hfdata = hfdata.take(split)

# multipeocwssing configs
worker_info = get_worker_info()
num_workers = worker_info.num_workers or 1
worker_id = worker_info.id or 0

world_size = get_world_size()
process_rank = get_rank()


class ImageDataset(IterableDataset):
    def __init__(self, dataset=hfdata):
        super().__init__()
        self.dataset = dataset
        self.worker_info = get_worker_info()

    def __iter__(self):
        for item in self.dataset:
            image = read_image(item["image"])
            class_label = item["label"]

            image = torch.tensor(image, dtype=config.dtype)
            class_label = torch.tensor(class_label, dtype=config.dtype)

            yield image, class_label


class ParallelImageDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset = hfdata):
        super().__init__()
        self.dataset = dataset
        self.worker_info = get_worker_info()

    def __len__(self):
        return split

    def __iter__(self):
        data_sampler = DistributedSampler(
            self.dataset,
            num_replicas=(num_workers * world_size),
            rank=(process_rank * num_workers + worker_id),
            shuffle=False,
        )

        for sample in iter(data_sampler):
            image = read_image(sample["image"])
            class_label = sample["label"]

            image = torch.tensor(image, dtype=config.dtype)
            class_label = torch.tensor(class_label, dtype=config.dtype)

            yield image, class_label
