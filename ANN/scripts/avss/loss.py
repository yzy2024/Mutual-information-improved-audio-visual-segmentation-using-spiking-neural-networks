import torch
import torch.nn as nn
import torch.nn.functional as F


def F10_IoU_BCELoss(pred_mask, ten_gt_masks, gt_temporal_mask_flag):
    """
    binary cross entropy loss (iou loss) of the total ten frames for multiple sound source segmentation

    Args:
    pred_mask: predicted masks for a batch of data, shape:[bs*10, N_CLASSES, 224, 224]
    ten_gt_masks: ground truth mask of the total ten frames, shape: [bs*10, 224, 224]
    """
    assert len(pred_mask.shape) == 4
    if ten_gt_masks.shape[1] == 1:
        ten_gt_masks = ten_gt_masks.squeeze(1)

    loss = nn.CrossEntropyLoss(reduction='none')(
        pred_mask, ten_gt_masks)  # [bs*10, 224, 224]
    loss = loss.mean(-1).mean(-1)  # [bs*10]
    loss = loss * gt_temporal_mask_flag  # [bs*10]
    loss = torch.sum(loss) / torch.sum(gt_temporal_mask_flag)

    return loss


def Mix_Dice_loss(pred_mask, norm_gt_mask, gt_temporal_mask_flag):
    """dice loss for aux loss

    Args:
        pred_mask (Tensor): (bs, 1, h, w)
        five_gt_masks (Tensor): (bs, 1, h, w)
    """
    assert len(pred_mask.shape) == 4
    pred_mask = torch.sigmoid(pred_mask)

    pred_mask = pred_mask.flatten(1)
    gt_mask = norm_gt_mask.flatten(1)
    a = (pred_mask * gt_mask).sum(-1)
    b = (pred_mask * pred_mask).sum(-1) + 0.001
    c = (gt_mask * gt_mask).sum(-1) + 0.001
    d = (2 * a) / (b + c)
    loss = 1 - d
    loss = loss * gt_temporal_mask_flag
    loss = torch.sum(loss) / torch.sum(gt_temporal_mask_flag)
    return loss

def mine_dv_loss(mi_map, seg_label, T=5, eps=1e-6):
    B = seg_label.shape[0]
    device = mi_map.device
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


def IouSemanticAwareLoss(pred_masks, mask_feature, mi_map, gt_mask, gt_temporal_mask_flag, weight_dict, **kwargs):
    total_loss = 0
    loss_dict = {}

    # iou_loss = weight_dict['iou_loss'] * \
    #     F10_IoU_BCELoss(pred_masks, gt_mask, gt_temporal_mask_flag)
    # total_loss += iou_loss
    # loss_dict['iou_loss'] = iou_loss.item()

    # mask_feature = torch.mean(mask_feature, dim=1, keepdim=True)
    # mask_feature = F.interpolate(
    #     mask_feature, gt_mask.shape[-2:], mode='bilinear', align_corners=False)
    # one_mask = torch.ones_like(gt_mask)
    # norm_gt_mask = torch.where(gt_mask > 0, one_mask, gt_mask)
    # mix_loss = weight_dict['mix_loss'] * \
    #     Mix_Dice_loss(mask_feature, norm_gt_mask, gt_temporal_mask_flag)
    # total_loss += mix_loss
    # loss_dict['mix_loss'] = mix_loss.item()

    # return total_loss, loss_dict

    iou_loss = F10_IoU_BCELoss(pred_masks, gt_mask, gt_temporal_mask_flag)
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
