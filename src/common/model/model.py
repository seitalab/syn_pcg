import sys
from argparse import Namespace
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.model.mae_transformer import mae_vit_base
from common.model.transformer import LinearEmbed, Transformer

def prepare_mae_model(params: Namespace) -> nn.Module:
    """
    Args:
        params (Namespace):
    Returns:
        predictor (nn.Module):
    """
    if params.modelname == "mae_base":
        model = mae_vit_base(params)
    else:
        raise NotImplementedError(
            f"{params.modelname} not available")
    return model

def prepare_clf_model(params: Namespace) -> nn.Module:
    """
    Args:
        params (Namespace):
    Returns:
        predictor (nn.Module):
    """
    is_mae = False
    if params.modelname == "mae_base":
        model_backbone = mae_vit_base(params)
        emb_dim = params.emb_dim
        foot = None
        is_mae = True

    elif params.modelname == "resnet18":
        from common.model.resnet import ResNet18
        foot = None
        emb_dim = None
        params.backbone_out_dim = params.clf_fc_dim
        model_backbone = ResNet18(params)    

    elif params.modelname == "resnet34":
        from common.model.resnet import ResNet34
        foot = None
        emb_dim = None
        params.backbone_out_dim = params.clf_fc_dim
        model_backbone = ResNet34(params)   

    elif params.modelname == "resnet50":
        from common.model.resnet import ResNet50
        foot = None
        emb_dim = None
        params.backbone_out_dim = params.clf_fc_dim
        model_backbone = ResNet50(params)   

    elif params.modelname == "effnetb1":
        sys.path.append("../common")
        from common.model.efficient_net import effnet1d_b1
        foot = None
        emb_dim = None
        seqlen = int(params.max_duration * params.target_freq)
        effnet_params = {
            "num_lead": params.num_lead,
            "sequence_length": seqlen,
            "backbone_out_dim": params.clf_fc_dim
        }        
        model_backbone = effnet1d_b1(**effnet_params)

    elif params.modelname == "effnetb0":
        sys.path.append("../common")
        from common.model.efficient_net import effnet1d_b0
        foot = None
        emb_dim = None
        seqlen = int(params.max_duration * params.target_freq)
        effnet_params = {
            "num_lead": params.num_lead,
            "sequence_length": seqlen,
            "backbone_out_dim": params.clf_fc_dim
        }        
        model_backbone = effnet1d_b0(**effnet_params)

    elif params.modelname == "lstm":
        from common.model.bi_lstm import VarDepthLSTM
        foot = LinearEmbed(params, add_cls_token=False)
        emb_dim = params.emb_dim
        params.backbone_out_dim = params.clf_fc_dim
        model_backbone = VarDepthLSTM(params,  params.emb_dim)

    elif params.modelname == "gru":
        from common.model.bi_gru import VarDepthGRU
        foot = LinearEmbed(params, add_cls_token=False)
        emb_dim = params.emb_dim
        params.backbone_out_dim = params.clf_fc_dim
        model_backbone = VarDepthGRU(params, params.emb_dim)

    elif params.modelname == "transformer":
        emb_dim = params.emb_dim
        foot = LinearEmbed(params)
        params.backbone_out_dim = params.clf_fc_dim
        params.feat_select = params.select_type
        model_backbone = Transformer(params)

    elif params.modelname == "mega":
        from common.model.mega import Mega
        emb_dim = params.emb_dim
        foot = LinearEmbed(params)
        params.backbone_out_dim = params.clf_fc_dim
        params.feat_select = params.select_type
        model_backbone = Mega(params)

    elif params.modelname == "s4":
        from common.model.s4 import S4
        emb_dim = params.emb_dim
        foot = LinearEmbed(params)
        params.backbone_out_dim = params.clf_fc_dim
        params.feat_select = params.select_type
        model_backbone = S4(params)

    else:
        raise NotImplementedError(
            f"{params.modelname} not available")


    if params.clf_mode == "logistic_regression":
        model = Classifier(model_backbone, params.emb_dim)
    else:
        head = HeadModule(params.clf_fc_dim)
        model = Predictor(
            model_backbone, 
            head, 
            emb_dim, 
            params.clf_fc_dim, 
            is_mae=is_mae,
            select_type=params.select_type,
            foot=foot
        )
    return model

class Classifier(nn.Module):

    def __init__(self, mae, emb_dim):
        super(Classifier, self).__init__()

        self.mae = mae
        self.fc = nn.Linear(emb_dim, 1)

    def forward(self, x):

        h, _, _ = self.mae.forward_encoder(x, mask_ratio=0) # (bs, num_chunks, emb_dim)
        h = self.fc(h[:, 0]) # use cls_token.
        return h

class Predictor(nn.Module):

    def __init__(
        self, 
        backbone: nn.Module, 
        head: nn.Module,
        emb_dim: Optional[int],
        backbone_out_dim: int,
        is_mae: bool=False,
        select_type: str="cls_token",
        foot: Optional[nn.Module]=None
    ) -> None:
        super(Predictor, self).__init__()

        self.backbone = backbone
        self.head = head

        self.is_mae = is_mae
        if is_mae:
            self.fc = nn.Linear(emb_dim, backbone_out_dim)

        self.select_type = select_type
        self.foot = foot

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Tensor of size 
                (batch_size, num_lead, seq_len).
        Returns:
            h (torch.Tensor): Tensor of size (batch_size, num_classes)
        """
        if self.is_mae:

            h, _, _ = self.mae.forward_encoder(x, mask_ratio=0) # (bs, num_chunks, emb_dim)

            if self.select_type == "cls_token":
                h = h[:, 0]
            elif self.select_type == "mean":
                h = torch.mean(h, dim=1)
            else:
                raise NotImplementedError
            h = self.fc(h)
        else:
            if self.foot is not None:
                x = self.foot(x)
            h = self.backbone(x)
        
        h = self.head(h)
        return h

class HeadModule(nn.Module):

    def __init__(self, in_dim: int):
        super(HeadModule, self).__init__()

        self.fc1 = nn.Linear(in_dim, 32)
        self.drop1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Tensor of size (num_batch, in_dim).
        Returns:
            feat (torch.Tensor): Tensor of size (num_batch, 1).
        """
        feat = F.relu(self.fc1(x))
        feat = self.drop1(feat)
        feat = self.fc2(feat)
        return feat