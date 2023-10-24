import os
import torch
from .data_utils import trivial_batch_collator, worker_init_reset_seed, pair_batch_collator, feat_batch_collator

datasets = {}
def register_dataset(name):
   def decorator(cls):
       datasets[name] = cls
       return cls
   return decorator

def make_dataset(name, is_training, split, features_object, **kwargs):
   """
       A simple dataset builder
   """
   dataset = datasets[name](is_training, split, features_object, **kwargs)
   return dataset

def make_data_loader(dataset, is_training, generator, batch_size, num_workers):
    """
        A simple dataloder builder
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=feat_batch_collator if "feats" in dataset.get_attributes()["dataset_name"] else pair_batch_collator,
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        shuffle=is_training,
        drop_last=is_training,
        generator=generator,
        persistent_workers=True
    )
    return loader
