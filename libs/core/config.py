import yaml
import os

DEFAULTS = {
    # random seed for reproducibility, a large number is preferred
    "init_rand_seed": 1234567891,
    # dataset loader, specify the dataset here
    "dataset_name": "sf_frame_dataset",
    "devices": ['cuda:0'], # default: single gpu
    "train_split": 'train',
    "test_split": 'test',
    "model_name": "slowfast-mlp",
    "dataset": {
        # number of frames for each feat
        "num_frames": 32,
        # default fps, may vary across datasets; Set to none for read from json file
        "fps": None,
    },
    "loader": {
        "batch_size": 8,
        "num_workers": 4,
    },
    # network architecture
    "model": {
        # type of backbone (convTransformer | conv)
        "backbone_type": 'conv',
        # type of FPN (fpn | identity)
        "fpn_type": "identity",
        "backbone_arch": (3, 3, 2),
        # scale factor between pyramid levels
        "scale_factor": 2,
        # number of heads in self-attention
        "n_head": 4,
        # window size for self attention; <=1 to use full seq (ie global attention)
        "n_mha_win_size": -1,
        # kernel size for embedding network
        "embd_kernel_size": 3,
        # (output) feature dim for embedding network
        "embd_dim": 512,
        # if attach group norm to embedding network
        "embd_with_ln": True,
        # feat dim for FPN
        "fpn_dim": 512,
        # if add ln at the end of fpn outputs
        "fpn_with_ln": True,
        # feat dim for head
        "head_dim": 512,
        # kernel size for reg/cls/center heads
        "head_kernel_size": 3,
        # number of layers in the head (including the final one)
        "head_num_layers": 3,
        # if attach group norm to heads
        "head_with_ln": True,
        # disable abs position encoding (added to input embedding)
        "use_abs_pe": False,
        # use rel position encoding (added to self-attention)
        "use_rel_pe": False,
    },
    "train_cfg": {
        "loss_weight": 1.0, 
        "init_loss_norm": 2000,
        "train_droppath" : 0.1,
        # gradient cliping, not needed for pre-LN transformer
        "clip_grad_l2norm": -1,
        # dropout ratios for tranformers
        "dropout": 0.0,
        # ratio for drop path
        "droppath": 0.1,
        # if to use label smoothing (>0.0)
        "label_smoothing": 0.0,
    },
    "test_cfg": {
    },
    # optimizer (for training)
    "opt": {
        # solver
        "type": "AdamW", # SGD or AdamW
        # solver params
        "momentum": 0.9,
        "weight_decay": 0.00001,
        "learning_rate": 0.001,
        "backbone_learning_rate":1e-6, 
        "head_learning_rate": 1e-3,
        # excluding the warmup epochs
        "epochs": 30,
        # lr scheduler: cosine / multistep
        "warmup": True,
        "warmup_epochs": 5,
        "schedule_type": "cosine",
        # in #epochs excluding warmup
        "schedule_steps": [],
        "schedule_gamma": 0.1,
    }
}

def _merge(src, dst):
    for k, v in src.items():
        if k in dst:
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            dst[k] = v

def load_default_config():
    config = DEFAULTS
    return config

def _update_config(config):
    # fill in derived fields
    config["model"]["input_dim"] = config["dataset"]["input_dim"]
    config["model"]["train_cfg"] = config["train_cfg"]
    config["model"]["test_cfg"] = config["test_cfg"]
    return config

def load_config(config_file, defaults=DEFAULTS):
    with open(config_file, "r") as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    
    # Model path is relative to config
    config["test_cfg"]["ckpt_path"] = os.path.join(os.path.dirname(config_file), config["test_cfg"]["ckpt_path"])

    _merge(defaults, config)
    config = _update_config(config)
    return config
