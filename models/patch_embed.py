import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=96, patch_size=8, in_chans=3, embed_dim=384) -> None:
        super().__init__()
        if patch_size <= 0 or img_size % patch_size != 0:
            raise ValueError(f"{img_size} is not divisible by {patch_size}")

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.grid_size = self.img_size // self.patch_size
        self.num_patches = self.grid_size**2

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(Batch, Channels, Height, Width) -> (Batch, # of Patches 'N', (patch*patch)*Channels)"""
        if x.shape[2:] != (self.img_size, self.img_size) or x.shape[1] != (self.in_chans):
            raise ValueError(f"x parameter is incorrect shape: expected (Batch, {self.in_chans}, {self.img_size}, {self.img_size})")
        # (B, C, H, W) -> (B, C, grid, patch, grid, patch) where grid * patch = img_size and img_size = H = W
        x = x.reshape(x.shape[0], self.in_chans, self.grid_size, self.patch_size, self.grid_size, self.patch_size)
        # (B, C, g, p, g, p) -> (B, g, g, p, p, C) i.e rotating/regrouping our prev operation
        x = x.permute(0, 2, 4, 3, 5, 1)
        # flatten grid, grid into 'N' num of patches and patch, patch, Channels into one shape representing the patch content
        x = x.reshape(x.shape[0], self.num_patches, self.patch_size**2 * self.in_chans)
        return x

