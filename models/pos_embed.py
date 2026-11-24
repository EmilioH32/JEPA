import torch

def _sincos_1d(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """pos: any shape. Returns (M, embed_dim), where M = pos.numel()"""
    if embed_dim % 2 != 0:
        raise ValueError(f"expected even embedding dimension: Given {embed_dim}")
    half = embed_dim // 2
    # 1 / 10000 ^ (i / half) for i in range(half)
    omega = 1 / 10000**(torch.arange(half).to(torch.float32) / half)
    # flatten position tensor 
    # i.e. (3, 4) -> (12)
    pos = pos.reshape(-1).to(torch.float32)
    # broadcasting for elementwise mult
    # i.e. n = 1 for mxn * nxp = mxp when I orginally had m, p
    angles = pos[:, None] * omega[None, :]
    # combining sine and cosine tensors into one matrix
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)