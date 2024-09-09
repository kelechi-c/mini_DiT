# pytorch related imports
from accelerate import accelerator
from torch import optim
import torch.nn.functional as func_nn
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

# generics and utilities
from .configs import train_configs
from .dit_model.vanilla_dit import Snowflake_DiT
from .utils import count_params
from ptflops import get_model_complexity_info
import re

dit_model = Snowflake_DiT()


def get_model_info(model):
    macs, params = get_model_complexity_info(
        model, (4, 32, 32), as_strings=True, print_per_layer_stat=True, verbose=True
    )
    flops = eval(re.findall(r"([\d.]+)", macs)[0]) * 2
    flops_unit = re.findall(r"([A-Za-z]+)", macs)[0][0]

    print("Computational complexity: {:<8}".format(macs))
    print("Computational complexity: {} {}Flops".format(flops, flops_unit))
    print("Number of parameters: {:<8}".format(params))


def trainer(model=dit_model, epochs=train_configs.epochs):
    pass
