import os
import copy
import random
import numpy as np
import random
import torch
import pytorchvideo.transforms.functional

class SpatialVideoCrop(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self,
            size : int, spatial_idx : int 
    ):
        super().__init__()
        self._size = size
        self._spatial_idx = spatial_idx

    def forward(self, frames: torch.Tensor):
        cropped_frames = pytorchvideo.transforms.functional.uniform_crop(
            frames, self._size, self._spatial_idx
        )
        return cropped_frames



class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self,
            alpha = 4
    ):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


def pair_batch_collator(batch):
    """
        A batch collator
    """
    ref_batch =  [x["ref_frames"] for x in batch]

    ref_slow = [x[0] for x in ref_batch]
    ref_fast = [x[1] for x in ref_batch]

    B =  len(ref_batch)
    C = 3
    T_slow = ref_slow[0].shape[1]
    T_fast = ref_fast[0].shape[1]
    H = ref_slow[0].shape[2]
    W = ref_slow[0].shape[3]

    ref_slow_e = torch.empty(B, C, T_slow, H, W)
    ref_fast_e = torch.empty(B, C, T_fast, H, W)

    for i in range(B):
        ref_slow_e[i] = ref_slow[i]
    
    for i in range(B):
        ref_fast_e[i] = ref_fast[i]
    
    ref_slow = ref_slow_e
    ref_fast = ref_fast_e


    ref_batch = [ref_slow, ref_fast]

    dis_batch =  [x["dis_frames"] for x in batch]
    dis_slow = [x[0] for x in dis_batch]
    dis_fast = [x[1] for x in dis_batch]
    dis_filenames = [x["dis"] for x in batch]

    dis_slow_e = torch.empty(B, C, T_slow, H, W)
    dis_fast_e = torch.empty(B, C, T_fast, H, W)

    for i in range(B):
        dis_slow_e[i] = dis_slow[i]
        dis_fast_e[i] = dis_fast[i]

    dis_slow = dis_slow_e
    dis_fast = dis_fast_e

    # dis_slow = torch.stack(dis_slow)
    # dis_fast = torch.stack(dis_fast)

    dis_batch = [dis_slow, dis_fast]
    
    mos_batch =  [x["mos"] for x in batch]
    mos_batch = torch.stack(mos_batch)

    return ref_batch, dis_batch, mos_batch, dis_filenames


def feat_batch_collator(batch):
    """
        A batch collator
    """
    feat_batch =  [x["feat"] for x in batch]
    mask_batch = [x["mask"] for x in batch]
    sample_name = [x["sample_name"] for x in batch]

    B = len(feat_batch)
    T, C = feat_batch[0].shape

    feats = torch.empty(B, T, C)
    masks = torch.empty(B,1,C)

    for i in range(B):
        feats[i] = feat_batch[i]
        masks[i]=  mask_batch[i]
    
    # mos_batch =  [x["mos"] for x in batch]
    # mos_batch = torch.stack(mos_batch)

    return feats, masks, sample_name





def trivial_batch_collator(batch):
    """
        A batch collator that does nothing
    """
    return batch

def worker_init_reset_seed(worker_id):
    """
        Reset random seed for each worker
    """
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

