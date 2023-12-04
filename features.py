# similar to feat_extraction.py
import argparse
import os.path

from parse import parse
import argparse
from easydict import EasyDict as edict
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Union
from pathlib import Path
from pprint import pprint

import torch
from dataset import VideoDataSet

def load_pairs(pairs_path: str)-> List[Tuple[str, str]]:
    """
    load pairs from file and return a list of tuples
    where each tuple is a pair of (ref, dis)

    Args:
        pairs_path (str): path to pairs file
    Returns:
        data_pairs (list): list of tuples
    """
    with open(pairs_path, "r") as f:
        data_pairs = f.readlines()
    data_pairs = [line.strip().split(",") for line in data_pairs]
    basedir = os.path.dirname(pairs_path)
    data_pairs = [(os.path.join(basedir, ref), os.path.join(basedir, dis)) for ref, dis in data_pairs]
    return data_pairs


def load_slowfast_configuration(path_to_config: str) -> edict: 
    """
    Load the configuration for the slowfast feature extractor directly
    from the tridivb_slowfast_feature_extractor package.

    Args:
        path_to_config (str): path to the configuration file

    Returns:
        cfg (dict): `updated` configuration dictionary
    """
    # path_to_config = "/home/azureuser/cloudfiles/code/Users/rubenal/tools_av_models/mlvideocodec/tools/mlc_vqa_e2e/tridivb_slowfast_feature_extractor/configs/SLOWFAST_8x8_R50.yaml"
    
    with open(path_to_config, "r") as stream:
        cfg = edict(yaml.safe_load(stream))
        cfg.DATA.VID_FILE_EXT = ".yuv"
        cfg.DATA.IN_FPS = 30
        cfg.DATA.OUT_FPS = 30
        cfg.DATA.SAMPLE_SIZE = [1920, 1080]
        cfg.DATA.YUV_WIDTH = 1920
        cfg.DATA.YUV_HEIGHT = 1080
        # add mean and std to cfg.DATA
        cfg.DATA.MEAN = [0.45, 0.45, 0.45]
        cfg.DATA.STD = [0.225, 0.225, 0.225]
        cfg.DATA.REVERSE_INPUT_CHANNEL = False
        cfg.MODEL.MODEL_NAME = "slowfast_r101"
        cfg.MODEL.SINGLE_PATHWAY_ARCH = [
            "2d",
            "c2d",
            "i3d",
            "slow",
            "x3d",
            "mvit",
            "maskmvit",
        ]
        cfg.MODEL.MULTI_PATHWAY_ARCH = ["slowfast"]
        cfg.DATA.STRIDE = 16

        cfg.DATA_LOADER.NUM_WORKERS = 1
        cfg.DATA_LOADER.PIN_MEMORY = True
        cfg.DATA_LOADER.NUM_GPUS = 1

    return cfg


def load_vmaf_configuration(path_to_config: str) -> edict:
    """
    Load the configuration for the vmaf feature extractor

    The arguments are the following:

    format can be one of:
        - yuv420p, yuv422p, yuv444p (8-Bit YUV)
        - yuv420p10le, yuv422p10le, yuv444p10le (10-Bit little-endian YUV)
        - yuv420p12le, yuv422p12le, yuv444p12le (12-Bit little-endian YUV)
        - yuv420p16le, yuv422p16le, yuv444p16le (16-Bit little-endian YUV)
    `width` and `height` are the width and height of the videos, in pixels
    `reference_path` and `distorted_path` are the paths to the reference and distorted video files
    `output_format` can be one of:
        text
        xml
        json

    Args:
        path_to_config (str): path to the configuration file
        args (dict): dictionary with the reference and distorted video paths

    Returns:
        cfg (dict): configuration dictionary
    """
    with open(path_to_config, "r") as stream:
        cfg = edict(yaml.safe_load(stream))

    # Model path is relative to config
    cfg.MODEL_PATH = os.path.join(os.path.dirname(path_to_config), cfg.MODEL_PATH)
    return cfg


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def load_slowfast_model(cfg) -> torch.nn.Module:
    """
    Load the SlowFast model and replace the final layer with a identity layer.

    Args:
        cfg (CfgNode): configs. Details can be found in yaml
    """
    # Load the video model and print model statistics.
    torch.hub._validate_not_a_forked_repo = (
        lambda a, b, c: True
    )

    # download the model
    model = torch.hub.load(
        "facebookresearch/pytorchvideo", model=cfg.MODEL.MODEL_NAME, pretrained=True
    )

    # Replace the final layers with a identity layer. Just return the previous input
    model.blocks[-1] = Identity()
    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.parallel.DataParallel(model, device_ids=list(range(cfg.NUM_GPUS)), dim=0)
    else:
        model = model.cpu()
    # Enable eval mode.
    model.eval()
    return model


@torch.no_grad()
def slowfast_inference(loader, model) -> np.ndarray:
    """
    Perform mutli-view that samples a segment of frames from a video
    and extract features from a pre-trained model.
    Args:
        loader (loader): video loader.
        model (model): the pretrained video model to test.
    """
    feat_arr = tuple()
    
    # for every consecutive pair of sampled frames (inputs) in the video:
    for inputs in tqdm(loader):
        # Transfer the data to the current GPU device.
        if isinstance(inputs, (list,)):
            for i in range(len(inputs)):
                # check if GPU is available
                if torch.cuda.is_available():
                    inputs[i] = inputs[i].cuda(non_blocking=True)
                else:
                    inputs[i] = inputs[i].cpu()
        else:
            # check if GPU is available
            if torch.cuda.is_available():
                inputs = inputs.cuda(non_blocking=True)
            else:
                inputs = inputs.cpu()

        feat = model(inputs)

        # adaptive average pooling across the w, h, t axis
        feat = torch.nn.AdaptiveAvgPool3d((1, 1, 1))(feat).squeeze()

        # when the last batch has only one sample
        if len(feat.shape) == 1:
            # squeeze operation converts (1, 2304) to (2304)
            # input arrays must have same number of dimensions for np.concatenate
            feat = feat.unsqueeze(0) # (2304) --> (1, 2304)
        feat = feat.cpu().numpy()
        feat_arr += (feat,)

    # concat all feats
    feat_arr = np.concatenate(feat_arr, axis=0)
    return feat_arr


def load_vmaf_model(cfg: edict) -> List[str]:
    """
    Wrapper to configure the VMAF feature extractor. We do not
    load any model here, but we just configure the command to
    run the VMAF feature extractor. This is done to keep the
    same interface as the other model-based feature extractors.

    For vmaf we fill out this template using the cfg file:

    PYTHONPATH=python ./python/vmaf/script/run_vmaf.py \
        yuv420p 576 324 \
        src01_hrc00_576x324.yuv \
        src01_hrc01_576x324.yuv \
        --out-fmt json
        --model vmaf_v0.6.1.pkl

    Args:
        cfg (edict): configuration dictionary

    Returns:
        cmd (List[str]): list of strings with the command to run the 
            VMAF feature extractor. The reference and distorted video
            paths are added later in the vmaf_inference function.
    """
    cmd = [
        cfg.RUN_VMAF_PATH,
        cfg.FORMAT,
        str(cfg.WIDTH),
        str(cfg.HEIGHT),
        None,
        None,
        "--out-fmt",
        cfg.OUTPUT_FORMAT,
        "--model",
        cfg.MODEL_PATH,
    ]

    return cmd

def _update_model(model: List[str], loader: List[str]):
    """
    Update the vmaf 'model' with the reference and distorted video paths

    Args:
        model (List[str]): list of strings with the command to run the
            VMAF feature extractor
        loader (List[str]): list of strings with the reference and distorted
            video paths

    Returns:
        model (List[str]): list of strings with the command to run the
            VMAF feature extractor with the reference and distorted video paths
    """
    model[4] = loader[0]
    model[5] = loader[1]
    return model
        

def vmaf_inference(loader, model):
    """
    Wrapper to run the VMAF feature extractor
    """
    import subprocess
    import json

    # model is a list of strings and the reference and distorted
    # video paths are added here
    model = _update_model(model, loader)

    print(f'Running command:\n{" ".join(model)}')
    # run the process and wait for it to finish
    result = subprocess.run(model, stdout=subprocess.PIPE, check=True)
    result = result.stdout.decode("utf-8")
    result = json.loads(result)
    return result


def load_vmaf_dataloader(data_pairs) -> List[str]:
    """
    Wrapper interface to keep the same interface as the other
    model-based feature extractors.

    Args:
        data_pairs (list): list of tuples containing the path to the
            reference and distorted videos.
    
    Returns:
        list: list of tuples containing the path to the
            reference and distorted videos.
    """
    return data_pairs


def load_slowfast_dataloader(cfg: edict, video: Path) -> torch.utils.data.DataLoader:
    """
    Load the video loader for the slowfast model.
    Args:
        cfg (CfgNode): configs. Details can be found in yaml
        video (Path): path to the video
    """
    video_dataset = VideoDataSet(cfg, video.parent, video.stem, read_vid_file=True)

    loader = torch.utils.data.DataLoader(
                video_dataset,
                batch_size=cfg.TEST.BATCH_SIZE,
                shuffle=False,
                sampler=None,
                num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                pin_memory=cfg.DATA_LOADER.PIN_MEMORY * cfg.NUM_GPUS,
                drop_last=False,
            )

    return loader

def extract_features(cfgs: List[edict], feature_extractors: List[str], video_pairs: List[Tuple[str, str]]):
    """
    Orchestrates the feature extraction process and creates the output object with shape:
    
    features_object = [
            {
                'feature_extractor': 'slowfast', 
                'output': {
                    'ref': np.array([1,2,3,4,5,6,7,8,9,10, ...]), 
                    'dis': np.array([1,2,3,4,5,6,7,8,9,10, ...])
                    },
                'sample_id': 0,
                'sample_name': '0442d8bdf9902226bfb38fbe039840d4f8ebe5270eda39d7dba56c2c3ae5becc'
            },
            {
                'feature_extractor': 'vmaf',
                'output': [{
                    "frameNum": 0,
                    "integer_adm2": 0.0,
                    "integer_adm_scale0": 0.0,
                    ...
                    },
                    ...
                    ],
                'sample_id': 0,
                'sample_name': '0442d8bdf9902226bfb38fbe039840d4f8ebe5270eda39d7dba56c2c3ae5becc'
            },
            ...
        ]
    
    Where sample_id is the index of the video pair in the video_pairs list. All features
    sharing the same sample_id correspond to the same video pair. The sample_name is the
    concatenated name of the reference and distorted video, separated by an underscore.
    
    Args:
        cfgs: list of configuration dictionaries
        feature_extractors: list of feature extractors to use
        video_pairs: list of tuples with (reference, distorted) video paths
    """
    slow_fast_cfg = cfgs[0]
    vmaf_cfg = cfgs[1]
    features_object = []

    for sample_id, video_pair in enumerate(video_pairs):
        
        # slowfast section
        slowfast_features = []
        for video in video_pair:

            video = Path(video)
            
            # load feature extractor
            slowfast_model = load_slowfast_model(slow_fast_cfg)

            # load dataloader
            slowfast_loader = load_slowfast_dataloader(slow_fast_cfg, video)

            # run inference
            slowfast_features.append(slowfast_inference(slowfast_loader, slowfast_model))

        ref = slowfast_features[0]
        dis = slowfast_features[1]

        # save features
        features_object.append({
            'feature_extractor': 'slowfast',
            'output': {
                'ref': ref,
                'dis': dis
            },
            'sample_id': sample_id,
            'sample_name': f"{video_pair[0]}_{video_pair[1]}"
        })

        # vmaf section
        vmaf_model = load_vmaf_model(vmaf_cfg)

        # load vmaf dataloader
        vmaf_loader = load_vmaf_dataloader(video_pair)

        # run inference
        vmaf_features = vmaf_inference(vmaf_loader, vmaf_model)

        # save features
        features_object.append({
            'feature_extractor': 'vmaf',
            'output': vmaf_features['frames'],
            'sample_id': sample_id,
            'sample_name': f"{video_pair[0]}_{video_pair[1]}"
        })

    return features_object
