# pytorch related imports
from jax._src.dtypes import dtype
import torch
from accelerate import accelerator
from torch import mode, optim
from torchsummary import summary
import torch.nn.functional as func_nn
from torch.cuda.amp import GradScaler

# generics and utilities
from .configs import train_configs
from .dit_model.vanilla_dit import Snowflake_DiT
from .utils import count_params, seed_everything
from .data.dataloader import ImageDataset
from ptflops import flops_to_string, get_model_complexity_info
import re
import wandb
import os
from time import time
import gc

#####
# get model summary and info
seed_everything()
dit_model = Snowflake_DiT()

model_summary = summary(dit_model, [(32, 32, 4), 1, 1])
print(model_summary)


def get_modelflops_info(model):
    macs, params = get_model_complexity_info(
        model, (4, 32, 32), as_strings=True, print_per_layer_stat=True, verbose=True
    )
    flops = eval(re.findall(r"([\d.]+)", macs)[0]) * 2
    flops_unit = re.findall(r"([A-Za-z]+)", macs)[0][0]

    print("Computational complexity: {:<8}".format(macs))
    print("Computational complexity: {} {}Flops".format(flops, flops_unit))
    print("Number of parameters: {:<8}".format(params))


flops_info = get_modelflops_info(dit_model)
print(f"Flops info: {flops_to_string}")

######
# training configs, callbacks, and optimizers
optimizer = optim.AdamW(dit_model.parameters(), lr=train_configs)
loss_fn = nn.
scaler = GradScaler()
wandb.login()
train_run = wandb.init(project="snowflake_dit", name="vit_1")
wandb.watch(dit_model, log_freq=100)


# training loop
def trainer(model=dit_model, epochs=train_configs.epochs):
    start_time = time()
    pass


# initilaize wandb


if os.path.exists(config.model_outpath) is not True:
    os.mkdir(config.model_outpath)

output_path = os.path.join(os.getcwd(), config.model_outpath)


def training_loop(model=vit, train_loader=train_loader, epochs=epochs, config=config):
    model.train()
    train_loss = 0.0

    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()

        torch.cuda.empty_cache()
        print(f"Training epoch {epoch+1}")

        for x, (image, label) in tqdm(enumerate(train_loader)):
            image = image.to(config.device)
            label = label.to(config.device)

            # every iterations
            torch.cuda.empty_cache()
            gc.collect()

            # Mixed precision training
            with torch.autocast(device_type="cuda", dtype=torch.float32):
                output = model(image)
                train_loss = criterion(output, label.long())
                train_loss = train_loss / config.grad_acc_step  # Normalize the loss

            # Scales loss. Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(train_loss).backward()

            if (x + 1) % config.grad_acc_step == 0:
                # Unscales the gradients of optimizer's assigned params in-place

                scaler.step(optimizer)
                # Updates the scale for next iteration
                scaler.update()
                optimizer.zero_grad()

            wandb.log({"loss": train_loss})

        print(f"Epoch {epoch} of {epochs}, train_loss: {train_loss.item():.4f}")

        print(f"Epoch @ {epoch} complete!")

    print(
        f"End metrics for run of {epochs}, train_loss: {train_loss.item():.4f}")

    save_model(model, config.safetensor_file)

    torch.save(model.state_dict(), f"{config.model_file}"))


training_loop()


print("snowflake, diffusion transformer training complete")
# Sayonara, time for model bending.