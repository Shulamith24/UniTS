"""
UniTS
"""
import math
import torch
import torch.nn.functional as F
from torch import nn

def random_masking(self, x, min_mask_ratio, max_mask_ratio):
    """
    Perform per-sample random masking.
    """
    N, V, L, D = x.shape  # batch, var, length, dim

    # Calculate mask ratios and lengths to keep for each sample in the batch
    mask_ratios = torch.rand(N, device=x.device) * \
        (max_mask_ratio - min_mask_ratio) + min_mask_ratio
    len_keeps = (L * (1 - mask_ratios)).long()

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    # ascend: small is keep, large is remove
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)

    # Create a range tensor and compare with len_keeps for mask generation
    range_tensor = torch.arange(L, device=x.device).expand(N, L)
    mask = (range_tensor >= len_keeps.unsqueeze(1))

    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    mask = mask.float()

    return mask

def right_masking(self, x, min_mask_ratio, max_mask_ratio):
    N, V, L, D = x.shape  # batch, var, length, dim

    # Randomly choose a mask ratio for each sample within the specified range
    mask_ratios = torch.rand(N, device=x.device) * \
        (max_mask_ratio - min_mask_ratio) + min_mask_ratio
    len_keeps = (L * (1 - mask_ratios)).long()

    # Binary mask creation without a for loop
    len_keeps_matrix = len_keeps.unsqueeze(1).expand(N, L)
    indices = torch.arange(L, device=x.device).expand_as(len_keeps_matrix)
    mask = indices >= len_keeps_matrix
    mask = mask.float()

    return mask

def choose_masking(self, x, right_prob, min_mask_ratio, max_mask_ratio):
    # Generate a random number to decide which masking function to use
    if torch.rand(1).item() > right_prob:
        return self.random_masking(x, min_mask_ratio, max_mask_ratio)
    else:
        return self.right_masking(x, min_mask_ratio, max_mask_ratio)


def get_mask_seq(self, mask, seq_len):
    mask_seq = mask.unsqueeze(dim=-1).repeat(1, 1, self.patch_len)
    mask_seq = mask_seq.permute(0, 2, 1)
    mask_seq = mask_seq.masked_fill(mask_seq == 0, -1e9)
    # Fold operation
    mask_seq = torch.nn.functional.fold(mask_seq, output_size=(
        seq_len, 1), kernel_size=(self.patch_len, 1), stride=(self.stride, 1))
    # Apply threshold to bring back to 0/1 values
    mask_seq = (mask_seq > 0).float()
    mask_seq = mask_seq.squeeze(dim=-1).squeeze(dim=1)
    return mask_seq