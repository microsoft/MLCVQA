import argparse
import os
import json
import numpy as np
from typing import List, Tuple
from features import extract_features, load_slowfast_configuration, load_vmaf_configuration, load_pairs
from eval import evaluate
from pprint import pprint
from preprocess import process_video_pairs

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--ref", type=str, default="./../../data/test_yuv/orig_480x360_30fps.yuv", help="path to reference video")
    args.add_argument("--dis", type=str, default="./../../data/test_yuv/comp1000_480x360_30fps.yuv", help="path to distorted video")
    args.add_argument("--dataset", type=str, default=None, help="path to pairs file, comma separated format")
    args.add_argument('--mlcvqa_config', type=str, metavar='DIR', help='path to a config file')
    args.add_argument("--slowfast_config", type=str, default="./tridivb_slowfast_feature_extractor/configs/SLOWFAST_8x8_R50.yaml", help="path to the slowfast config file")
    args.add_argument("--vmaf_config", type=str, default="./configs/vmaf_config.yaml", help="path to the vmaf config file")
    args.add_argument("--preprocess", action="store_true", help="input videos should be 1920x1080, 10sec, 30fps")
    args.add_argument("--temp_folder", type=str, help="location for the preprocessed videos if --preprocess is set to True.")

    args = args.parse_args()

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
    
    # add preprocessing to make sure the input is 1080p, 30fps, 10sec
    if args.preprocess:
        if args.temp_folder is None:
            print("please provide a temp folder")
            exit()
        if not os.path.exists(args.temp_folder):
            os.makedirs(args.temp_folder)
        data_pairs = process_video_pairs(data_pairs, args.temp_folder)

        
    slowfast_cfg_path = args.slowfast_config
    vmaf_cfg_path = args.vmaf_config

    slow_fast_cfg = load_slowfast_configuration(slowfast_cfg_path)
    vmaf_cfg = load_vmaf_configuration(vmaf_cfg_path)

    cfgs = [slow_fast_cfg, vmaf_cfg]

    features_object = extract_features(cfgs, feature_extractors, data_pairs)

    results = evaluate(args, features_object)

    pprint(results)

    #TODO: remove the temp folder