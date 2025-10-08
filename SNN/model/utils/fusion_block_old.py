# import torch
# import torch.nn as nn


# class CrossModalMixer(nn.Module):
#     def __init__(self, dim=256, n_heads=8, qkv_bias=False, dropout=0.):
#         super().__init__()

#         self.dim = dim
#         self.n_heads = n_heads
#         self.dropout = dropout
#         self.scale = (dim // n_heads)**-0.5

#         self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
#         self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(dropout)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(dropout)

#     def forward(self, feature_map, audio_feature):
#         """channel attention for modality fusion

#         Args:
#             feature_map (Tensor): (bs, c, h, w)
#             audio_feature (Tensor): (bs, 1, c)

#         Returns:
#             Tensor: (bs, c, h, w)
#         """
#         # print(feature_map.shape, audio_feature.shape)
#         flatten_map = feature_map.flatten(2).transpose(1, 2)
#         B, N, C = flatten_map.shape

#         q = self.q_proj(audio_feature).reshape(
#             B, 1, self.n_heads, C // self.n_heads).permute(0, 2, 1, 3)
#         kv = self.kv_proj(flatten_map).reshape(
#             B, N, 2, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
#         k, v = kv.unbind(0)
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)
#         x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
#         x = self.proj_drop(self.proj(x))

#         x = x.sigmoid()
#         fusion_map = torch.einsum('bchw,bc->bchw', feature_map, x.squeeze())
#         # print(fusion_map.shape)
#         return fusion_map


# def build_fusion_block(type, **kwargs):
#     if type == 'CrossModalMixer':
#         return CrossModalMixer(**kwargs)
#     else:
#         raise ValueError

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalMixer(nn.Module):
    def __init__(self, dim=256, n_heads=8, qkv_bias=False, dropout=0.):
        super().__init__()

        self.dim = dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.scale = (dim // n_heads)**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, feature_map, audio_feature):
        """channel attention for modality fusion

        Args:
            feature_map (Tensor): (bs, c, h, w)
            audio_feature (Tensor): (bs, 1, c)

        Returns:
            Tensor: (bs, c, h, w)
        """
        flatten_map = feature_map.flatten(2).transpose(1, 2)
        B, N, C = flatten_map.shape

        q = self.q_proj(audio_feature).reshape(
            B, 1, self.n_heads, C // self.n_heads).permute(0, 2, 1, 3)
        kv = self.kv_proj(flatten_map).reshape(
            B, N, 2, self.n_heads, C // self.n_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        x = self.proj_drop(self.proj(x))

        x = x.sigmoid()
        fusion_map = torch.einsum('bchw,bc->bchw', feature_map, x.squeeze())
        return fusion_map

class MINE(nn.Module):
    def __init__(self, hidden_dim=128):

        super(MINE, self).__init__()
        self.fc1 = nn.Linear(512, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

        # self.bn1 = nn.BatchNorm1d(hidden_dim)
        # self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, y):
        """
        x, y: [B * H * W, C]
        return: scalar MI estimate per pair [B * H * W]
        """
        joint = torch.cat([x, y], dim=1)  # [B * H * W, 2C]

        x = self.fc1(joint)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x.squeeze(-1)  # [B * H * W]


class MI(nn.Module):
    def __init__(self):
        super().__init__()
        self.mine = MINE()
        self.proj = nn.Linear(256, 256)      # 补充投影层
        self.proj_drop = nn.Dropout(p=0.1) 

    def forward(self, feature_map, audio_feature):
        B, C, H, W = feature_map.shape
        visual = feature_map.permute(0, 2, 3, 1).reshape(B * H * W, C)
        
        audio_feature = audio_feature.squeeze(1)
        audio = audio_feature.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W)
        audio = audio.permute(0, 2, 3, 1).reshape(B * H * W, C)

        mi_values = self.mine(visual, audio)
        mi_map = mi_values.reshape(B, H, W)
        mi_map = mi_map.view(B, 1, H * W)
        feature_map_flat = feature_map.view(B, C, H * W)
        x = (mi_map @ feature_map_flat.transpose(1, 2)).reshape(B, 1, C)
        x = self.proj_drop(self.proj(x))

        x = x.sigmoid()
        fusion_map = torch.einsum('bchw,bc->bchw', feature_map, x.squeeze())
        mi_map = mi_map.reshape(B, 1, H, W)  # reshape MI map to [B, 1, H, W]
          # Normalize the MI map to [0, 1]
        return fusion_map, mi_map



def build_fusion_block(type, **kwargs):
    if type == 'CrossModalMixer':
        return CrossModalMixer(**kwargs)
    elif type == 'MINE':
        return MI(**kwargs)
    else:
        raise ValueError
