import os
import shutil
import time
import json
import pickle
from typing import Dict
from statistics import mean, stdev
import numpy as np
from scipy import stats
import torch


def filter_results(results_dict):
    """
    Given a batch of results, filter them by their sample_name
    and return the mean of the predictions

    Args:
        results_dict (Dict): A dictionary containing the results

    Returns:
        Dict: A dictionary containing the filtered and aggregated results
    """
    filtered_results = {}
    for i, sample_name in enumerate(results_dict["sample_name"]):
        if sample_name not in filtered_results.keys():
            filtered_results[sample_name] = [results_dict["pred_mos"][i]]
        else:
            filtered_results[sample_name].append(results_dict["pred_mos"][i])

    final_dict = {"sample_name": [], "pred_mos": []}

    for sample_name, pred_lis in filtered_results.items():
        final_dict["sample_name"].append(sample_name)
        final_dict["pred_mos"].append(np.mean(pred_lis))

    return final_dict
    


def evaluate(results_dict):
    avg_preds = results_dict["pred_mos"]
    avg_gts = results_dict["gt_mos"]
    try:
        srocc , srcc_pval  = stats.spearmanr(avg_preds, avg_gts)
        plcc , plcc_pval  = stats.pearsonr(avg_preds, avg_gts)
        std = np.std(avg_preds)
        return srocc, srcc_pval, plcc, plcc_pval, std
    except Exception as e:
        print(e)
        return -1.0, -1.0, -1.0, -1.0, -1.0
    
