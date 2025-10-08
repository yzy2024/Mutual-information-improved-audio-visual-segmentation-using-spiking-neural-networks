import torch
import torch.nn as nn
import time
import os
from .backbone import build_backbone
# from .neck import build_neck
from .head import build_head
from .vggish import VGGish

import logging
def getLogger(log_file, name, fmt='%(asctime)s %(levelname)s ==> %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
log_name = time.strftime('%Y%m%d-%H%M%S', time.localtime())
log_file='/home/songyu/vit-llm/yzy/MI_AVS/work_dir'
dir_name="print"
if not os.path.exists(os.path.join(log_file, dir_name)):
    os.mkdir(os.path.join(log_file, dir_name))
log_file = os.path.join(log_file, dir_name, f'{log_name}.log')
logger = getLogger(log_file, __name__)
class AVSegFormer(nn.Module):
    def __init__(self,
                 backbone,
                 vggish,
                 head,
                 neck=None,
                 audio_dim=128,
                 embed_dim=256,
                 T=5,
                 freeze_audio_backbone=True,
                 *args, **kwargs):
        super().__init__()

        self.embed_dim = embed_dim
        self.T = T
        self.freeze_audio_backbone = freeze_audio_backbone
        self.backbone = build_backbone(**backbone)
        self.vggish = VGGish(**vggish)
        self.head = build_head(**head)
        self.audio_proj = nn.Linear(audio_dim, embed_dim)

        if self.freeze_audio_backbone:
            for p in self.vggish.parameters():
                p.requires_grad = False
        self.freeze_backbone(True)

        self.neck = neck
        # if neck is not None:
        #     self.neck = build_neck(**neck)
        # else:
        #     self.neck = None

    def freeze_backbone(self, freeze=False):
        for p in self.backbone.parameters():
            p.requires_grad = not freeze

    def mul_temporal_mask(self, feats, vid_temporal_mask_flag=None):
        if vid_temporal_mask_flag is None:
            return feats
        else:
            if isinstance(feats, list):
                out = []
                for x in feats:
                    out.append(x * vid_temporal_mask_flag)
            elif isinstance(feats, torch.Tensor):
                out = feats * vid_temporal_mask_flag

            return out

    def extract_feat(self, x):
        feats = self.backbone(x)
        if self.neck is not None:
            feats = self.neck(feats)
        return feats

    def forward(self, audio, frames, vid_temporal_mask_flag=None):
        if vid_temporal_mask_flag is not None:
            vid_temporal_mask_flag = vid_temporal_mask_flag.view(-1, 1, 1, 1)
        with torch.no_grad():
            audio_feat = self.vggish(audio)  # [B*T,128]

        audio_feat = audio_feat.unsqueeze(1)
        audio_feat = self.audio_proj(audio_feat)
        img_feat = self.extract_feat(frames)
        img_feat = self.mul_temporal_mask(img_feat, vid_temporal_mask_flag)
        pred, mask_feature, mi_map = self.head(img_feat, audio_feat)
        pred = self.mul_temporal_mask(pred, vid_temporal_mask_flag)
        mask_feature = self.mul_temporal_mask(
            mask_feature, vid_temporal_mask_flag)

        return pred, mask_feature, mi_map
