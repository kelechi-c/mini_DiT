import torch
from torch import nn
from .dit_blocks import DiTBlock, FinalMlp
from timm.models.vision_transformer import PatchEmbed

# main diffusion transformer block


class Snowflake_DiT(nn.Module):
    def __init__(
        self,
        embed_dim=384,
        latent_size=32,
        num_layers=12,
        patch_size=4,
        channels=4,
        classes=1000,
        c_dropout=0.1,
        mlp_ratio=4,
        learn_sigma=True,  # covariance
    ):
        super().__init__()
        # base attributes
        self.in_channels = channels  # channels for noised latent
        self.out_channels = (
            channels * 2
        )  # doubling do it can be split to noise/covariance
        self.patch_size = patch_size  # image patch size
        self.num_classes = classes  # class count, 1000 for imagenet-1k
        self.learn_covariance = (
            learn_sigma  # to learn the model/distribution covariance
        )

        self.layer_norm = nn.LayerNorm(embed_dim)

        # patch embedding layer
        self.image_embedder = PatchEmbed(
            latent_size, patch_size=patch_size, in_chans=channels, embed_dim=embed_dim
        )
        self.patch_count = self.patchify.num_patches

        # timstep/class embedding
        self.time_embedder = None
        self.class_embedder = None

        # dit blocks
        dit_blocks = [DiTBlock() for _ in range(num_layers)]
        self.dit_layers = nn.Sequential(*dit_blocks)

        # 'Linear and Reshaoe' block
        self.final_layer = FinalMlp(
            embed_dim=embed_dim, patch_size=patch_size, out_channels=4
        )

    def forward(
        self, x_img: torch.Tensor, y_class: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        # patchify images and apply positional encoding
        img_patches = self.image_embedder(x_img)
        time_embed, class_embed = (
            self.time_embedder(timestep),  # timestep
            self.class_embedder(y_class),  # image class conditioning
        )
        conditioning = time_embed + class_embed  # merge timestep with class label

        # diffusion transformer blocks, attention
        output = self.dit_layers(img_patches, conditioning)
        # output liear projection
        output = self.final_linear(output, conditioning)

        pred_noise, covariance = output.chunk(2, dim=1)

        return pred_noise, covariance
