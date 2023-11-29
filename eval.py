import argparse
import os
import glob
import time
from pprint import pprint
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, fix_random_seed


def load_mlcvqa_configuration(args: dict)-> dict:
    """
    Loads the configuration file from the provided path.

    Args:
        config_path (str): path to the configuration file

    Returns:
        cfg (dict): configuration file
    """
    if os.path.isfile(args.mlcvqa_config):
        cfg = load_config(args.mlcvqa_config)
    else:
        raise ValueError("Config file does not exist.")
    return cfg


def load_mlcvqa_checkpoint(ckpt_path: str)-> str:
    """
    Loads and 'validates' the checkpoint file from the provided path. 
    If the path is a folder, it will load the last checkpoint file.

    Args:
        ckpt_path (str): path to the checkpoint file

    Returns:
        ckpt_file (str): path to the checkpoint file
    """
    if ".pth.tar" in ckpt_path:
        assert os.path.isfile(ckpt_path), "CKPT file does not exist!"
        ckpt_file = ckpt_path
    else:
        assert os.path.isdir(ckpt_path), "CKPT file folder does not exist!"
        ckpt_file_list = sorted(glob.glob(os.path.join(ckpt_path, '*.pth.tar')))
        ckpt_file = ckpt_file_list[-1]
    return ckpt_file


def load_mlcvqa_dataloader(cfg: dict, features)-> torch.utils.data.DataLoader:
    """
    Loads the dataloader for the provided configuration file.

    Args:
        cfg (dict): configuration file
        features (dict): features dictionary

    Returns:
        val_loader (torch.utils.data.DataLoader): mlcvqa dataloader
    """
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], features, **cfg['dataset']
    )

    val_loader = make_data_loader(
         dataset = val_dataset,
         is_training = False,
         generator = None,
         batch_size = cfg['test_loader']['batch_size'],
         num_workers = cfg['test_loader']['num_workers']
        )
    return val_loader


def load_mlcvqa_model(cfg: dict)-> nn.DataParallel:
    """
    Loads the model for the provided configuration file and checkpoint file.

    Args:
        cfg (dict): configuration file
        ckpt_file (str): path to the checkpoint file

    Returns:
        model (nn.DataParallel): mlcvqa model
    """
    ckpt_path = cfg['test_cfg']['ckpt_path']
    ckpt_file = load_mlcvqa_checkpoint(ckpt_path)

    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model = nn.DataParallel(model, device_ids=cfg['devices'])

    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location = lambda storage, loc: storage.cuda(cfg['devices'][0])
    )
    # load ema model instead
    model.load_state_dict(checkpoint['state_dict_ema'])
    model = model.cuda()
    return model


def prepare_output_file(cfg: dict)-> str:
    """
    Prepares the output path file for the results.

    Args:
        cfg (dict): configuration file

    Returns:
        output_file (str): path to the output file
    """
    output_folder = Path(cfg['test_cfg']['save_outputs'])
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / 'mlc_vqa_results.json'
    return output_file


def evaluate(config_path: str, features_object)-> tuple:
    """
    Computes mlcvqa predictions from given feature pairs

    Args:
        config_path (str): path to the configuration file
        ckpt_file (str): path to the checkpoint file
        
    Returns:
        sample_names (list): video pairs
        pred_mos (list): predicted MOS
    """
    
    cfg = load_mlcvqa_configuration(config_path)

    val_loader = load_mlcvqa_dataloader(cfg, features_object)
    
    model = load_mlcvqa_model(cfg)
    
    output_file = None
    if cfg['test_cfg']['save_outputs']:
        output_file = prepare_output_file(cfg)
    
   
    start = time.time()
    sample_names, pred_mos = valid_one_epoch(
        val_loader,
        model,
        -1,
        output_file=output_file,
        print_freq=cfg['test_cfg']['print_freq']
    )
    end = time.time()
    
    print("Sample names: ", sample_names, "predicted MOS: ", pred_mos)
    print("All done! Total time: {:0.2f} sec".format(end - start))
    return sample_names, pred_mos


def mock_features_object(path: str):
        """
        Mock the features object. We need to create a mock object that has: 
        features_object = [
            {
                'feature_extractor': 'slowfast', 
                'output': {
                    'ref': np.array([1,2,3,4,5,6,7,8,9,10]), 
                    'dis': np.array([1,2,3,4,5,6,7,8,9,10])
                    },
                'sample_id': 0
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
                'sample_id': 0
            },
            ...
        ]

        Args:
            path (str): path to mock vmaf features

        Returns:
            features_object (tuple): tuple of dictionaries
        """

        # VmafFeatureExtractor
        import json
        import numpy as np
        # list all json files
        json_frames = []
        json_name = None
        sample_id = 0
        features_object = ()
        
        for file in os.listdir(path):
            if file.endswith(".json"):
                json_file = os.path.join(path, file)
                with open(json_file, 'r') as fid:
                    data = json.load(fid)
                    json_frames = data['frames']
                    json_name = data['asset']['identifier']
            else:
                continue
                
            if len(json_frames) > 300:
                continue
                
            # SlowFeatureExtractor
            # create a numpy array of size (18,2304) for every json file
            ref = np.random.rand(18,2304)
            dis = np.random.rand(18,2304)
            
            features_object += (
                {
                    'feature_extractor': 'slowfast',
                    'output': {
                        'ref': ref,
                        'dis': dis
                    },
                    'sample_id': sample_id,
                    'sample_name': json_name
                },
                {
                    'feature_extractor': 'vmaf',
                    'output': json_frames,
                    'sample_id': sample_id,
                    'sample_name': json_name
                }
            )
            sample_id += 1

        return features_object


