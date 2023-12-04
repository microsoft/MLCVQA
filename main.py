import argparse
import os
import json
import numpy as np
from typing import List, Tuple
from features import extract_features, load_slowfast_configuration, load_vmaf_configuration, load_pairs
from eval import evaluate
from pprint import pprint
from preprocess import run_preprocess
import shutil


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str, help="path to reference video")
    parser.add_argument("--dis", type=str, help="path to distorted video")
    parser.add_argument("--dataset", type=str, help="path to pairs file, comma separated format")
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    parser.add_argument('--mlcvqa_config', type=str,
                        default=os.path.join(config_dir, "mlcvqa_config.yaml"),
                        help='path to a config file')
    parser.add_argument("--slowfast_config", type=str,
                        default=os.path.join(config_dir, "SLOWFAST_8x8_R50.yaml"),
                        help="path to the slowfast config file")
    parser.add_argument("--vmaf_config", type=str,
                        default=os.path.join(config_dir, "vmaf_config.yaml"),
                        help="path to the vmaf config file")
    parser.add_argument("--preprocess", action="store_true", help="input videos should be 1920x1080, 10sec, 30fps")

    args = parser.parse_args()

    data_pairs :List[Tuple[str, str]] = []
    if args.dataset is not None:
        # load pairs from dataset
        data_pairs = load_pairs(args.dataset)
    else:
        # load pairs from command line
        data_pairs = [(args.ref, args.dis)]

    # configure slowfast and vmaf feature extractors
    feature_extractors = [
        "slowfast",
        "vmaf",
    ]
        
    slowfast_cfg_path = args.slowfast_config
    vmaf_cfg_path = args.vmaf_config

    slow_fast_cfg = load_slowfast_configuration(slowfast_cfg_path)
    vmaf_cfg = load_vmaf_configuration(vmaf_cfg_path)

    cfgs = [slow_fast_cfg, vmaf_cfg]

    # add preprocessing to make sure the input is 1080p, 30fps, 10sec
    if args.preprocess:
        data_pairs, preprocessed_dir = run_preprocess(args, data_pairs)

    features_object = extract_features(cfgs, feature_extractors, data_pairs)

    results = evaluate(args, features_object)

    pprint(results)

    #remove output folder if it exists
    if args.preprocess:
        shutil.rmtree(preprocessed_dir, ignore_errors=True)