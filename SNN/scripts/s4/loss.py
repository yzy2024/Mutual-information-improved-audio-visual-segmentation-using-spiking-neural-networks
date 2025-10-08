# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# def F1_IoU_BCELoss(pred_masks, first_gt_mask):
#     """
#     binary cross entropy loss (iou loss) of the first frame for single sound source segmentation

#     Args:
#     pred_masks: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
#     first_gt_mask: ground truth mask of the first frame, shape: [bs, 1, 1, 224, 224]
#     """
#     assert len(pred_masks.shape) == 4
#     pred_masks = torch.sigmoid(pred_masks)  # [bs*5, 1, 224, 224]

#     indices = torch.tensor(list(range(0, len(pred_masks), 5)))
#     indices = indices.cuda()
#     first_pred = torch.index_select(
#         pred_masks, dim=0, index=indices)  # [bs, 1, 224, 224]
#     assert first_pred.requires_grad == True, "Error when indexing predited masks"
#     if len(first_gt_mask.shape) == 5:
#         first_gt_mask = first_gt_mask.squeeze(1)  # [bs, 1, 224, 224]

#     first_bce_loss = nn.BCELoss()(first_pred, first_gt_mask)

#     return first_bce_loss


# def F1_Dice_loss(pred_masks, first_gt_mask):
#     """dice loss for aux loss

#     Args:
#         pred_mask (Tensor): (bs*5, 1, h, w)
#         five_gt_masks (Tensor): (bs, 1, 1, h, w)
#     """
#     assert len(pred_masks.shape) == 4
#     pred_masks = torch.sigmoid(pred_masks)

#     indices = torch.tensor(list(range(0, len(pred_masks), 5)))
#     indices = indices.cuda()
#     first_pred = torch.index_select(
#         pred_masks, dim=0, index=indices)  # [bs, 1, 224, 224]
#     assert first_pred.requires_grad == True, "Error when indexing predited masks"
#     if len(first_gt_mask.shape) == 5:
#         first_gt_mask = first_gt_mask.squeeze(1)  # [bs, 1, 224, 224]

#     pred_mask = first_pred.flatten(1)
#     gt_mask = first_gt_mask.flatten(1)
#     a = (pred_mask * gt_mask).sum(-1)
#     b = (pred_mask * pred_mask).sum(-1) + 0.001
#     c = (gt_mask * gt_mask).sum(-1) + 0.001
#     d = (2 * a) / (b + c)
#     loss = 1 - d
#     return loss.mean()

# def mine_dv_loss(mi_map, seg_label, T=5, eps=1e-6):
#     B = seg_label.shape[0]
#     device = mi_map.device

#     # 1. 提取每个样本的第一帧 mi_map
#     indices = torch.arange(0, B * T, T, device=device)  # e.g., [0, 5, 10, ...]
#     first_frame_mi = mi_map.index_select(0, indices)  # shape: [B, 28, 28]

#     # 2. 下采样 seg_label 到 28x28（与 mi_map 对齐）
#     seg_label_down = F.avg_pool2d(seg_label.unsqueeze(1).float(), kernel_size=8).squeeze(1)  # [B, 28, 28]
#     positive_mask = (seg_label_down > 0.4).float()
#     negative_mask = 1.0 - positive_mask

#     # 3. 正负样本区域数量（避免除0）
#     num_pos = positive_mask.sum(dim=(1, 2)).clamp(min=1.0)
#     num_neg = negative_mask.sum(dim=(1, 2)).clamp(min=1.0)

#     # 4. 正样本期望 E_p[T]
#     Ep = ((first_frame_mi * positive_mask).sum(dim=(1, 2)) / num_pos)  # [B]

#     # 5. 负样本期望 log E_q[exp(T)]
#     Eq = (torch.log(torch.exp(first_frame_mi) * negative_mask + eps).sum(dim=(1, 2)) / num_neg)  # [B]

#     # 6. MINE 损失
#     mi_loss = -(Ep - Eq)  # [B]
#     return mi_loss.mean()

# def IouSemanticAwareLoss(pred_masks, mask_feature, gt_mask, weight_dict, loss_type='bce', **kwargs):
#     total_loss = 0
#     loss_dict = {}

#     if loss_type == 'bce':
#         loss_func = F1_IoU_BCELoss
#     elif loss_type == 'dice':
#         loss_func = F1_Dice_loss
#     else:
#         raise ValueError

#     iou_loss = loss_func(pred_masks, gt_mask)
#     total_loss += weight_dict['iou_loss'] * iou_loss
#     loss_dict['iou_loss'] = weight_dict['iou_loss'] * iou_loss.item()

#     # mask_feature = torch.mean(mask_feature, dim=1, keepdim=True)
#     # mask_feature = F.interpolate(
#     #     mask_feature, gt_mask.shape[-2:], mode='bilinear', align_corners=False)
#     # mix_loss = weight_dict['mix_loss']*loss_func(mask_feature, gt_mask)
#     # total_loss += mix_loss
#     # loss_dict['mix_loss'] = mix_loss.item()
#     # if 'mine_dv_loss' in weight_dict:
#     mine_loss = mine_dv_loss(mask_feature, gt_mask, **kwargs)
#     total_loss += weight_dict['mine_dv_loss'] * mine_loss
#     loss_dict['mine_dv_loss'] = weight_dict['mine_dv_loss'] * mine_loss.item()

#     return total_loss, loss_dict


import torch
import torch.nn as nn
import torch.nn.functional as F


def F1_IoU_BCELoss(pred_masks, first_gt_mask):
    """
    binary cross entropy loss (iou loss) of the first frame for single sound source segmentation

    Args:
    pred_masks: predicted masks for a batch of data, shape:[bs*5, 1, 224, 224]
    first_gt_mask: ground truth mask of the first frame, shape: [bs, 1, 1, 224, 224]
    """
    assert len(pred_masks.shape) == 4
    pred_masks = torch.sigmoid(pred_masks)  # [bs*5, 1, 224, 224]

    indices = torch.tensor(list(range(0, len(pred_masks), 5)))
    indices = indices.cuda()
    first_pred = torch.index_select(
        pred_masks, dim=0, index=indices)  # [bs, 1, 224, 224]
    assert first_pred.requires_grad == True, "Error when indexing predited masks"
    if len(first_gt_mask.shape) == 5:
        first_gt_mask = first_gt_mask.squeeze(1)  # [bs, 1, 224, 224]

    first_bce_loss = nn.BCELoss()(first_pred, first_gt_mask)

    return first_bce_loss


def F1_Dice_loss(pred_masks, first_gt_mask):
    """dice loss for aux loss

    Args:
        pred_mask (Tensor): (bs*5, 1, h, w)
        five_gt_masks (Tensor): (bs, 1, 1, h, w)
    """
    assert len(pred_masks.shape) == 4
    pred_masks = torch.sigmoid(pred_masks)

    indices = torch.tensor(list(range(0, len(pred_masks), 5)))
    indices = indices.cuda()
    first_pred = torch.index_select(
        pred_masks, dim=0, index=indices)  # [bs, 1, 224, 224]
    assert first_pred.requires_grad == True, "Error when indexing predited masks"
    if len(first_gt_mask.shape) == 5:
        first_gt_mask = first_gt_mask.squeeze(1)  # [bs, 1, 224, 224]

    pred_mask = first_pred.flatten(1)
    gt_mask = first_gt_mask.flatten(1)
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

    # 1. 提取每个样本的第一帧 mi_map
    indices = torch.arange(0, B * T, T, device=device)  # e.g., [0, 5, 10, ...]
    first_frame_mi = mi_map.index_select(0, indices).squeeze(1) # shape: [B, 128, 128]

    # 2. 下采样 seg_label 到 128x128（与 mi_map 对齐）
    seg_label_down = F.avg_pool2d(seg_label.squeeze(1).float(), kernel_size=4).squeeze(1)  # [B, 128, 128]
    # print(seg_label_down.shape)
    positive_mask = (seg_label_down > 0.4).float()
    negative_mask = 1.0 - positive_mask

    # 3. 正负样本区域数量（避免除0）
    num_pos = positive_mask.sum(dim=(1, 2)).clamp(min=1.0)
    num_neg = negative_mask.sum(dim=(1, 2)).clamp(min=1.0)

    # 4. 正样本期望 E_p[T]
    Ep = ((first_frame_mi * positive_mask).sum(dim=(1, 2)) / num_pos)  # [B]

    # 5. 负样本期望 log E_q[exp(T)]
    Eq = (torch.log(torch.exp(first_frame_mi) * negative_mask + eps).sum(dim=(1, 2)) / num_neg)  # [B]

    # 6. MINE 损失
    mi_loss = -(Ep - Eq)  # [B]
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



