import os
import json, time
import numpy as np
import pandas as pd
import torch
import random
from torch.utils.data import Dataset
from torch.nn import functional as F
from yaml import load
from .datasets import register_dataset

@register_dataset("sf_vmaf_feats_dataset")
class SlowFastFeatDataset(Dataset):
    def __init__(self,
                is_training,
                split,
                features_object,
                input_dim,
                padded_seq_len,
                aug_dirs = [],
                n_temporal_aug =  0,
                num_frames=32,
                sampling_rate=2,
                fps = 30,
                alpha = 4,
                window_size = 64,
                stride = 16
                ):
        
        assert isinstance(split, str) or isinstance(split, list)


        self.is_training = is_training
        self.split = split
        self.input_dim = input_dim
        self.padded_seq_len =  padded_seq_len
        
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.fps = fps
        self.alpha = alpha
        self.n_temporal_aug = n_temporal_aug

        self.db_attributes = {
            'dataset_name': 'sf_clic_feats_n32_fps_30_samp_rate2',
        }

        self.window_size = window_size
        self.stride = stride
        self.desired_len = int(self.window_size/self.sampling_rate)
        self.vmaf_feat_names = [
            'VMAF_feature_adm2_score',
            'VMAF_feature_adm_scale0_score',
            'VMAF_feature_adm_scale1_score',
            'VMAF_feature_adm_scale2_score',
            'VMAF_feature_adm_scale3_score',
            'VMAF_feature_motion2_score',
            'VMAF_feature_motion_score',
            'VMAF_feature_vif_scale0_score',
            'VMAF_feature_vif_scale1_score',
            'VMAF_feature_vif_scale2_score',
            'VMAF_feature_vif_scale3_score'
        ]
        self.data_list = self._load_feature_db(features_object)

        
    def get_attributes(self):
        return self.db_attributes

    def _load_feature_db(self, features_object):
        """
        Load the feature database. It has shape:
        features_object = [
            {
                'feature_extractor': 'slowfast', 
                'output': {
                    'ref': np.array([1,2,3,4,5,6,7,8,9,10]), 
                    'dis': np.array([1,2,3,4,5,6,7,8,9,10])
                    },
                'sample_id': 0,
                'sample_name': 'ref_01_vs_dis_01'
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
                'sample_name': 'ref_01_vs_dis_01'
            },
            ...
        ]

        Args:
            features_object (list): list of dictionaries containing the features

        Returns:
            tuple: tuple containing a dictionary with the features
        """
        dict_db = tuple()
        # order the elements in the list by sample_id
        features_object = sorted(features_object, key=lambda k: k['sample_id'])

        # get all sample ids
        sample_ids = list(set([x['sample_id'] for x in features_object]))

        for sample_id in sample_ids:
            # get features corresponding to the current sample id
            features_object_sample = [x for x in features_object if x['sample_id'] == sample_id]
            if self.split == "train":
                # not implemented
                pass
            else:
                if self.n_temporal_aug == 0:
                    # temporarly set to 1
                    self.n_temporal_aug = 1

                for temp_aug_id in range(self.n_temporal_aug):
                    # this can be done in a more efficient way
                    # we add the slowfast features to the dict_db first and then we add the vmaf features
                    for sample in features_object_sample:
                        if sample['feature_extractor'] == 'slowfast':
                            dict_db += ({
                                'ref': sample['output']['ref'].astype(np.float32),
                                'dis' : sample['output']['dis'].astype(np.float32),
                                'sample_id' : sample['sample_id'],
                                'temp_aug_id' : temp_aug_id if self.n_temporal_aug > 1 else None,
                                'sample_name' : sample['sample_name'],
                            }, )
                    
                    assert len(dict_db) != 0, "dict_db is empty"

                    for sample in features_object_sample:
                        if sample['feature_extractor'] == 'vmaf':
                                # access the dict_db element corresponding to the current sample_id and temp_aug_id,
                                # add the vmaf features
                            for i, item in enumerate(dict_db):
                                if item['sample_id'] == sample['sample_id'] and item['temp_aug_id'] == temp_aug_id:
                                    item["vmaf_feat"] = self._load_vmaf_feature(sample['output']).astype(np.float32)
                                    break

                # set it back to 0
                if self.n_temporal_aug == 1:
                    self.n_temporal_aug = 0
        
        return dict_db


    def __len__(self):
        return len(self.data_list)


    def pad_and_get_mask(self, feat, padding_val=0.0):
        """
            Generate padded and masks from a list of dict items
        """
        feat_len = feat.shape[-1]
        padded_seq_len = self.padded_seq_len
        
        feat = torch.nn.functional.pad(feat, (0, self.padded_seq_len - feat_len), mode='constant', value=padding_val)
        
        mask = torch.arange(padded_seq_len)[None, :] < feat_len

        return feat, mask

    def _load_vmaf_feature(self, feature):
        """
        VMAF features are in the form of a list of dictionaries:
        [
            {
                "frameNum": 0,
                "integer_adm2": 0.0,
                ...,
            },
            ...
        ]
        We get the features from the frames and return a list of numpy arrays

        Args:
            features_object (list): list of dictionaries containing the features

        Returns:
            np.array: numpy array of numpy arrays containing the features with shape
            (18, 352)
        """
        # load vmaf features, list of dictionaries, every dictionary is a frame with
        # the features
        sample_vmaf_df = pd.DataFrame(feature)
        sample_vmaf_df = sample_vmaf_df[self.vmaf_feat_names]
        sample_vmaf_feat = []
        for i in range(0, (len(sample_vmaf_df)-self.stride), self.stride):
            tmp = sample_vmaf_df.loc[range(i, min((i+self.window_size-1), len(sample_vmaf_df)), self.sampling_rate)]
            if len(tmp)<self.desired_len:
                tmp = tmp.reindex(range(self.desired_len), fill_value=1e-5)
            sample_vmaf_feat.append(tmp.to_numpy().flatten())

        sample_vmaf_df = np.array(sample_vmaf_feat)
        return sample_vmaf_df


    def __getitem__(self, idx):
        curr_db = self.data_list[idx]
        ref_feats, dis_feats, vmaf_feat, sample_name = curr_db['ref'], curr_db['dis'], curr_db['vmaf_feat'], curr_db['sample_name']
        
        assert ref_feats.shape == (18,2304)
        assert dis_feats.shape == (18,2304)

        diff_feats =  ref_feats - dis_feats + 1e-9
        diff_feats = np.concatenate([dis_feats, diff_feats, vmaf_feat], axis = 1) # this should return a vector with (18, 352+2304)
        diff_feats = torch.tensor(diff_feats).permute(1,0)

        if self.n_temporal_aug > 1:
            if self.is_training:
                # not implemented
                pass
            else:
                start_idx = curr_db["temp_aug_id"]
                diff_feats = diff_feats[:,start_idx::self.n_temporal_aug]
            
        diff_feats, mask = self.pad_and_get_mask(diff_feats)

        data_dict = {
            'feat' : diff_feats,
            'mask' : mask,
            'sample_name': sample_name # apparently this is the id of the video, TODO check where and how it is used
            }

        return data_dict



