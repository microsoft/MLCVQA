from ast import Raise
import math

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import MaskedConv1D, Scale, LayerNorm


class MLP_Head(nn.Module):
    def __init__(
        self,
        layer_dims = [1024,512,128,1],
        dropout_probs = [0.3,0.3,0.3],
        input_dim = 2304
        ):

        super().__init__()
        self.mlp = nn.ModuleList()
        for i in range(len(layer_dims)):
            if i == 0:
                self.mlp.append(nn.Linear(input_dim, layer_dims[i]))
            else:
                self.mlp.append(nn.Linear(layer_dims[i-1], layer_dims[i]))
            if i < len(dropout_probs):
                self.mlp.append(nn.Dropout(dropout_probs[i]))

        self.relu = nn.ReLU()


    def forward(self, x):

        for i in range(len(self.mlp)):
            x =  self.mlp[i](x)
            if i != len(self.mlp) - 1:
                x = self.relu(x)
        
        return x
        

class RegHead(nn.Module):
    """
    Shared 1D Conv heads for regression
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        fpn_levels,
        recency_weighting = False,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.LeakyReLU,
        with_ln=False
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        # segment regression
        self.mos_head = MaskedConv1D(
                feat_dim, 1, kernel_size,
                stride=1, padding=kernel_size//2
            )
        self.recency_weighting = recency_weighting

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_mos = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_mos, _ = self.mos_head(cur_out, cur_mask)
            cur_mos = F.relu(self.scale[l](cur_mos))
            
            true_len = torch.sum(cur_mask[0,0]).item()
            cur_mos = cur_mos[:,:,:true_len]

            if self.recency_weighting:    
                W =  torch.linspace(1,3, steps= true_len).detach().cuda()
                cur_mos =  (cur_mos@W) / W.sum()
            else:
                cur_mos = cur_mos.mean(-1)
            
            #cur_mos,_ = F.relu(self.scale[l](cur_mos)).min(-1)
            #print("cm",cur_mos.shape)
            out_mos += (cur_mos, )
            
        out_mos = torch.hstack(out_mos).mean(-1)
        #print("om",out_mos.shape)
        # fpn_masks remains the same
        return out_mos


class RegVotesHead(nn.Module):
    """
    Shared 1D Conv heads for regression
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        fpn_levels,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size//2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        # segment regression
        self.mos_head = MaskedConv1D(
                feat_dim, 5, kernel_size,
                stride=1, padding=kernel_size//2
            )

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_mos = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_mos, _ = self.mos_head(cur_out, cur_mask)
            cur_mos = cur_mos[cur_mask]
            cur_mos = F.relu(self.scale[l](cur_mos)).mean(-1)

            out_mos += (cur_mos, )
            
        out_mos = torch.hstack(out_mos).mean(-1)

        return out_mos