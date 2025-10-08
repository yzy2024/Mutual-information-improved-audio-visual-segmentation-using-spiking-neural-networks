# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# def F5_IoU_BCELoss(pred_mask, five_gt_masks):
#     """
#     binary cross entropy loss (iou loss) of the total five frames for multiple sound source segmentation

#     Args:
#     pred_mask: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
#     five_gt_masks: ground truth mask of the total five frames, shape: [bs*5, 1, 224, 224]
#     """
#     assert len(pred_mask.shape) == 4
#     pred_mask = torch.sigmoid(pred_mask)  # [bs*5, 1, 224, 224]
#     # five_gt_masks = five_gt_masks.view(-1, 1, five_gt_masks.shape[-2], five_gt_masks.shape[-1]) # [bs*5, 1, 224, 224]
#     loss = nn.BCELoss()(pred_mask, five_gt_masks)

#     return loss


# def F5_Dice_loss(pred_mask, five_gt_masks):
#     """dice loss for aux loss

#     Args:
#         pred_mask (Tensor): (bs, 1, h, w)
#         five_gt_masks (Tensor): (bs, 1, h, w)
#     """
#     assert len(pred_mask.shape) == 4
#     pred_mask = torch.sigmoid(pred_mask)

#     pred_mask = pred_mask.flatten(1)
#     gt_mask = five_gt_masks.flatten(1)
#     a = (pred_mask * gt_mask).sum(-1)
#     b = (pred_mask * pred_mask).sum(-1) + 0.001
#     c = (gt_mask * gt_mask).sum(-1) + 0.001
#     d = (2 * a) / (b + c)
#     loss = 1 - d
#     return loss.mean()


# def IouSemanticAwareLoss(pred_mask, mask_feature, gt_mask, weight_dict, loss_type='bce', **kwargs):
#     total_loss = 0
#     loss_dict = {}

#     if loss_type == 'bce':
#         loss_func = F5_IoU_BCELoss
#     elif loss_type == 'dice':
#         loss_func = F5_Dice_loss
#     else:
#         raise ValueError

#     iou_loss = weight_dict['iou_loss'] * loss_func(pred_mask, gt_mask)
#     total_loss += iou_loss
#     loss_dict['iou_loss'] = iou_loss.item()

#     mask_feature = torch.mean(mask_feature, dim=1, keepdim=True)
#     mask_feature = F.interpolate(
#         mask_feature, gt_mask.shape[-2:], mode='bilinear', align_corners=False)
#     mix_loss = weight_dict['mix_loss']*loss_func(mask_feature, gt_mask)
#     total_loss += mix_loss
#     loss_dict['mix_loss'] = mix_loss.item()

#     return total_loss, loss_dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def F1_IoU_BCELoss(pred_mask, first_gt_mask):
    """
    binary cross entropy loss (iou loss) of the first frame for single sound source segmentation

    Args:
    pred_masks: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    first_gt_mask: ground truth mask of the first frame, shape: [bs, 1, 1, 224, 224]
    """
    assert len(pred_mask.shape) == 4
    pred_mask = torch.sigmoid(pred_mask)  # [bs*5, 1, 224, 224]
    # five_gt_masks = five_gt_masks.view(-1, 1, five_gt_masks.shape[-2], five_gt_masks.shape[-1]) # [bs*5, 1, 224, 224]
    loss = nn.BCELoss()(pred_mask, first_gt_mask)

    return loss


def F1_Dice_loss(pred_mask, five_gt_mask):
    """dice loss for aux loss

    Args:
        pred_mask (Tensor): (bs*5, 1, h, w)
        five_gt_masks (Tensor): (bs, 1, 1, h, w)
    """
    assert len(pred_mask.shape) == 4
    pred_mask = torch.sigmoid(pred_mask)

    pred_mask = pred_mask.flatten(1)
    gt_mask = five_gt_mask.flatten(1)
    a = (pred_mask * gt_mask).sum(-1)
    b = (pred_mask * pred_mask).sum(-1) + 0.001
    c = (gt_mask * gt_mask).sum(-1) + 0.001
    d = (2 * a) / (b + c)
    loss = 1 - d
    return loss.mean()

def mine_dv_loss(mi_map, seg_label, T=5, eps=1e-6):
    # print("mi_map.shape:", mi_map.shape)
    # print("seg_label.shape:", seg_label.shape)
    # mi_map.shape: torch.Size([30, 1, 128, 128])
    # seg_label.shape: torch.Size([6, 1, 1, 512, 512])
    B = seg_label.shape[0]
    device = mi_map.device

    # # 1. 提取每个样本的第一帧 mi_map
    # indices = torch.arange(0, B * T, T, device=device)  # e.g., [0, 5, 10, ...]
    # first_frame_mi = mi_map.index_select(0, indices).squeeze(1) # shape: [B, 128, 128]

    # # 2. 下采样 seg_label 到 128x128（与 mi_map 对齐）
    # seg_label_down = F.avg_pool2d(seg_label.squeeze(1).float(), kernel_size=4).squeeze(1)  # [B, 128, 128]
    # # print(seg_label_down.shape)
    # positive_mask = (seg_label_down > 0.4).float()
    # negative_mask = 1.0 - positive_mask

    # # 3. 正负样本区域数量（避免除0）
    # num_pos = positive_mask.sum(dim=(1, 2)).clamp(min=1.0)
    # num_neg = negative_mask.sum(dim=(1, 2)).clamp(min=1.0)

    # # 4. 正样本期望 E_p[T]
    # Ep = ((first_frame_mi * positive_mask).sum(dim=(1, 2)) / num_pos)  # [B]

    # # 5. 负样本期望 log E_q[exp(T)]
    # Eq = (torch.log(torch.exp(first_frame_mi) * negative_mask + eps).sum(dim=(1, 2)) / num_neg)  # [B]

    # # 6. MINE 损失
    # mi_loss = -(Ep - Eq)  # [B]
    # return mi_loss.mean()
    # Step 1: Downsample seg_label to 128x128 and reshape to [B*T, 128, 128]
    seg_label = seg_label.squeeze(1).float()  # [B, T, 512, 512]
    seg_label = F.avg_pool2d(
        seg_label.reshape(B, 1, 512, 512), kernel_size=4  # -> [B, 1, 128, 128]
    ).squeeze(1)  # -> [B, 128, 128]

    # Step 2: Prepare masks
    positive_mask = (seg_label > 0.4).float()  # [B*T, 128, 128]
    negative_mask = 1.0 - positive_mask

    # Step 3: Compute positive and negative sample count
    num_pos = positive_mask.sum(dim=(1, 2)).clamp(min=1.0)
    num_neg = negative_mask.sum(dim=(1, 2)).clamp(min=1.0)

    # Step 4: Compute E_p[T]
    mi_map = mi_map.squeeze(1)  # [B*T, 128, 128]
    Ep = ((mi_map * positive_mask).sum(dim=(1, 2)) / num_pos)  # [B*T]

    # Step 5: Compute log E_q[exp(T)]
    Eq = (torch.log(torch.exp(mi_map) * negative_mask + eps).sum(dim=(1, 2)) / num_neg)  # [B*T]

    # Step 6: Final loss
    mi_loss = -(Ep - Eq)  # [B*T]
    return mi_loss.mean()

def IouSemanticAwareLoss(pred_masks, mask_feature, mi_map, gt_mask, weight_dict, loss_type='bce', **kwargs):
    total_loss = 0
    loss_dict = {}

    if loss_type == 'bce':
        loss_func = F1_IoU_BCELoss
    elif loss_type == 'dice':
        loss_func = F1_Dice_loss
    else:
        raise ValueError

    iou_loss = loss_func(pred_masks, gt_mask)
    total_loss += weight_dict['iou_loss'] * iou_loss
    loss_dict['iou_loss'] = weight_dict['iou_loss'] * iou_loss.item()

    mask_feature = torch.mean(mask_feature, dim=1, keepdim=True)
    mask_feature = F.interpolate(
        mask_feature, gt_mask.shape[-2:], mode='bilinear', align_corners=False)
    # mix_loss = weight_dict['mix_loss']*loss_func(mask_feature, gt_mask)
    # total_loss += mix_loss
    # loss_dict['mix_loss'] = mix_loss.item()

    # if 'mine_dv_loss' in weight_dict:
    mine_loss = mine_dv_loss(mi_map, gt_mask, **kwargs)
    total_loss += weight_dict['mine_dv_loss'] * mine_loss
    loss_dict['mine_dv_loss'] = weight_dict['mine_dv_loss'] * mine_loss.item()

    return total_loss, loss_dict