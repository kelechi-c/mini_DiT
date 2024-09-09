import torch
from torch import nn
from .dit_blocks import DiTBlock, FinalMlp, TimestepEmbedder, LabelEmbedder
from timm.models.vision_transformer import PatchEmbed
from einops import rearrange


class Snowflake_DiT(nn.Module):
    """
    complete diffusion transformer for class-conditioned image generation.
    """

    def __init__(
        self,
        embed_dim=384,
        latent_size=32,
        num_layers=12,
        patch_size=4,
        channels=4,
        classes=1000,
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
        self.time_embedder = TimestepEmbedder()
        self.class_embedder = LabelEmbedder()

        self.pos_embed = nn.Parameter(
            torch.zeros(1, classes, embed_dim), requires_grad=False
        )

        # dit blocks
        self.dit_blocks = [DiTBlock() for _ in range(num_layers)]
        self.dit_layers = nn.Sequential(*self.dit_blocks)

        # 'Linear and Reshape' block
        self.final_layer = FinalMlp(
            embed_dim=embed_dim, patch_size=patch_size, out_channels=4
        )

        self.initialize_zero()

    def initialize_zero(self):
        """This section was majorly adapted from the official implementation"""
        self.apply(self._init_weights)

        # init label embeddder
        nn.init.normal_(self.class_embedder.codebook.weight, std=0.02)

        # initialize timestep embedder
        nn.init.normal_(self.time_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embedder.mlp[2].weight, std=0.02)

        # init for patchembed
        patch_emb_weight = self.image_embedder.proj.weight.data
        nn.init.xavier_uniform_(patch_emb_weight.view([patch_emb_weight.shape[0], -1]))
        nn.init.constant_(self.image_embedder.proj.bias, 0)

        # init for final linear decoder
        nn.init.constant_(self.final_layer.ada_ln[-1].weight, 0)
        nn.init.constant_(self.final_layer.ada_ln[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # Zero initialization for DiT blocks, implementing the papers 'adaLN-Zero' approach
        for layer in self.dit_blocks:
            nn.init.constant_(layer.adaLN[-1].weight, 0)
            nn.init.constant_(layer.adaLN[-1].bias, 0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

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
        # reverse the patching operation to return image
        final_image_latent = self.unpatchify(output)

        return final_image_latent  # (n, h, w, 2c)

    def unpatchify(self, x_patch: torch.Tensor):
        # patch_size from PatchEmbed
        p_size = self.image_embedder.patch_size[0]
        # reshape (batch_size, seq_len, [patch, patch, channels]) -> (batch, channels, height, patch_size, width)
        x = rearrange(x_patch, "n t (p p c) -> n c h p w", p=p_size)
        # reshape to multiply h/w with patch_size, resulting shape -> (batch, channels, height, width)
        x_img = rearrange(x, "n c h p w -> n c (h p) (w p)")

        return x_img
