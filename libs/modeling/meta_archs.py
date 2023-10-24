from ast import Raise
import math

import torch
from torch import nn
from torch.nn import functional as F

from .models import register_meta_arch, make_backbone, make_neck
from .blocks import MaskedConv1D, Scale, LayerNorm
from .heads import RegHead, RegVotesHead


@register_meta_arch("FeatConvTransformer")
class FeatTransformer(nn.Module):
    """
        Convs and Transformer
    """
    def __init__(
        self,
        backbone_type,         # a string defines which backbone we use
        fpn_type,              # a string defines which fpn we use
        backbone_arch,         # a tuple defines # layers in embed / stem / branch
        scale_factor,          # scale factor between branch layers
        input_dim,             # input feat dim
        padded_seq_len,        # max sequence length (used for training)
        n_head,                # number of heads for self-attention in transformer
        n_mha_win_size,        # window size for self attention; -1 to use full seq
        embd_kernel_size,      # kernel size of the embedding network
        embd_dim,              # output feat channel of the embedding network
        embd_with_ln,          # attach layernorm to embedding network
        fpn_dim,               # feature dim on FPN
        fpn_with_ln,           # if to apply layer norm at the end of fpn
        head_dim,              # feature dim for head
        head_num_layers,       # number of layers in the head (including the classifier)
        head_kernel_size,      # kernel size for reg/cls heads
        recency_weighting,     # whether to use recency weighting
        head_with_ln,          # attache layernorm to reg/cls heads
        use_abs_pe,            # if to use abs position encoding
        use_rel_pe,            # if to use rel position encoding
        train_cfg,             # other cfg for training
        test_cfg               # other cfg for testing
    ):
        super().__init__()
        self.scale_factor = scale_factor

        self.fpn_strides = [scale_factor**i for i in range(backbone_arch[-1]+1)]
        
        self.padded_seq_len = padded_seq_len

        if isinstance(n_mha_win_size, int):
            self.mha_win_size = [n_mha_win_size]*len(self.fpn_strides)
        else:
            assert len(n_mha_win_size) == len(self.fpn_strides)
            self.mha_win_size = n_mha_win_size
            
        max_div_factor = 1

        # training time config

        self.train_dropout = train_cfg['dropout']
        self.train_droppath = train_cfg['train_droppath']

        # backbone network: conv + transformer
        assert backbone_type in ['convTransformer', 'conv']
        if backbone_type == 'convTransformer':
            self.backbone = make_backbone(
                'convTransformer',
                **{
                    'n_in' : input_dim,
                    'n_embd' : embd_dim,
                    'n_head': n_head,
                    'n_embd_ks': embd_kernel_size,
                    'padded_seq_len': padded_seq_len,
                    'arch' : backbone_arch,
                    'mha_win_size': self.mha_win_size,
                    'scale_factor' : scale_factor,
                    'with_ln' : embd_with_ln,
                    'attn_pdrop' : 0.0,
                    'proj_pdrop' : self.train_dropout,
                    'path_pdrop' : self.train_droppath,
                    'use_abs_pe' : use_abs_pe,
                    'use_rel_pe' : use_rel_pe
                }
            )
        else:
            self.backbone = make_backbone(
                'conv',
                **{
                    'n_in': input_dim,
                    'n_embd': embd_dim,
                    'n_embd_ks': embd_kernel_size,
                    'arch': backbone_arch,
                    'scale_factor': scale_factor,
                    'with_ln' : embd_with_ln
                }
            )

        assert fpn_type in ['fpn', 'identity']

        self.neck = make_neck(
            fpn_type,
            **{
                'in_channels' : [embd_dim] * (backbone_arch[-1] + 1),
                'out_channel' : fpn_dim,
                'scale_factor' : scale_factor,
                'with_ln' : fpn_with_ln
            }
        )

        self.reg_head = RegHead(
            fpn_dim,
            head_dim,
            len(self.fpn_strides),
            recency_weighting = recency_weighting,
            kernel_size=head_kernel_size,
            num_layers=head_num_layers,
            with_ln=head_with_ln
        )
        if train_cfg["loss"] == "mse":
            self.reg_loss = nn.MSELoss()
        elif train_cfg["loss"] == "huber":
            self.reg_loss = nn.HuberLoss()
        elif train_cfg["loss"] == "mae":
            self.reg_loss = nn.L1Loss()
        else:
            raise Exception("loss needs to be either huber or MAE or MSE") 

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        #print(self.parameters())
        #print(list(set(p.device for p in self.parameters())))
        return list(set(p.device for p in self.parameters()))[0]

    def forward(self, data):

        # forward the network (backbone -> heads)
        # feats, batched_masks, gt_mos, file_names = data
        feats, batched_masks, sample_names = data

        feats =  feats.cuda()
        
        batched_masks = batched_masks.cuda().detach()

        # gt_mos = gt_mos.cuda()

        backbone_feats, masks = self.backbone(feats, batched_masks)
        
        fpn_feats, fpn_masks = self.neck(backbone_feats, masks)

        pred_score = self.reg_head(fpn_feats, fpn_masks)

        # losses = self.losses(
            # pred_score,
            # gt_mos
        # )
        # return pred_score, losses
        return pred_score

    def losses(
        self, pred_score, gt_mos
    ):
        reg_loss = self.reg_loss(pred_score, gt_mos)
        #sort_loss = torch_sort_loss(pred_score, gt_mos)
        losses = {"reg_loss": reg_loss,
                    #"sort_loss": sort_loss,
                    "final_loss": reg_loss}# + sort_loss}
        return losses
    