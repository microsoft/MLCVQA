import argparse
import os
import json
import numpy as np
from typing import List, Tuple
from features import extract_features, load_slowfast_configuration, load_vmaf_configuration, load_pairs
from eval import evaluate
from pprint import pprint
from preprocess import run_preprocess


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--ref", type=str, help="path to reference video")
    args.add_argument("--dis", type=str, help="path to distorted video")
    args.add_argument("--dataset", type=str, help="path to pairs file, comma separated format")
    args.add_argument('--mlcvqa_config', type=str, default="./configs/mlcvqa_config.yaml", help='path to mlcvqa config file')
    args.add_argument("--slowfast_config", type=str, default="./tridivb_slowfast_feature_extractor/configs/SLOWFAST_8x8_R50.yaml", help="path to the slowfast config file")
    args.add_argument("--vmaf_config", type=str, default="./configs/vmaf_config.yaml", help="path to the vmaf config file")
    args.add_argument("--preprocess", action="store_true", help="input videos should be 1920x1080, 10sec, 30fps")

    args = args.parse_args()
    print(type(args))

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
        print(f'Removing temporary folder: {preprocessed_dir}.')
        os.system(f'rm -rf {preprocessed_dir}')