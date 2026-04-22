import torch
import torch.nn.functional as F

def _pad_truncate_to(x: torch.Tensor, target_shape: torch.Size, pad_value: float = 0.0) -> torch.Tensor:
    """
    Pad (right) or truncate tensor x to match target_shape (including batch dim).
    Works for 2D/3D+ tensors; pads only on the *end* of each dimension.
    """
    assert x.dim() == len(target_shape), f"x.dim={x.dim()} vs target_shape={target_shape}"
    out = x

    slicers = []
    for d, tsize in enumerate(target_shape):
        csize = out.size(d)
        slicers.append(slice(0, min(csize, tsize)))
    out = out[tuple(slicers)]

    pads = []
    for d in reversed(range(out.dim())):
        csize = out.size(d)
        tsize = target_shape[d]
        add = max(tsize - csize, 0)
        pads.extend([0, add])  # pad only at the end
    if any(pads):
        out = F.pad(out, pads, mode="constant", value=pad_value)

    slicers = []
    for d, tsize in enumerate(target_shape):
        slicers.append(slice(0, tsize))
    return out[tuple(slicers)]


def _overlap_mask_like(targets: torch.Tensor, orig_shape: torch.Size) -> torch.Tensor:
    """
    Build a mask (same shape as targets) with 1's in the region that overlaps with the original
    model output (before padding), and 0's elsewhere (the padded/unavailable part).
    Only considers dims after the batch dim.
    """
    device = targets.device
    dtype  = targets.dtype
    mask = torch.zeros_like(targets, dtype=dtype, device=device)

    if targets.dim() == 2:
        B, Tt = targets.shape
        To = orig_shape[1]
        T = min(Tt, To)
        mask[:, :T] = 1
    elif targets.dim() == 3:
        B, Tt, Dt = targets.shape
        To = orig_shape[1]
        Do = orig_shape[2]
        T = min(Tt, To)
        D = min(Dt, Do)
        mask[:, :T, :D] = 1
    else:
        slicers = [slice(None)]
        for d in range(1, targets.dim()):
            slicers.append(slice(0, min(targets.size(d), orig_shape[d])))
        mask[tuple(slicers)] = 1

    return mask


def _broadcast_base_mask(base_mask: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Common case: base_mask is [B, T] and targets is [B, D, T].
    If already [B, D, T], it's returned as-is. Otherwise tries to broadcast.
    """
    if base_mask.shape == targets.shape:
        return base_mask

    # [B, T] -> [B, 1, T] -> [B, D, T]
    if (base_mask.ndim == targets.ndim - 1 and
        base_mask.shape[0] == targets.shape[0] and
        base_mask.shape[-1] == targets.shape[-1]):
        return base_mask.unsqueeze(1).expand(-1, targets.shape[1], -1)

    # Try generic broadcasting
    try:
        return base_mask + torch.zeros_like(targets)
    except RuntimeError:
        # Last resort: insert singleton dims until it broadcasts, then expand channels
        bm = base_mask
        while bm.ndim < targets.ndim:
            bm = bm.unsqueeze(1)
        if bm.shape[1] == 1 and targets.shape[1] != 1:
            bm = bm.expand(-1, targets.shape[1], *bm.shape[2:])
        return bm
