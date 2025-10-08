import torch
import torch.nn
import os
import sys
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from mmcv import Config
import argparse
from utils import pyutils
from utility import mask_iou, Eval_Fmeasure, save_mask
from utils.logger import getLogger
from model import build_model
from dataloader import build_dataset
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
def save_mi_visualization(mi_tensor, save_path, video_names):
    os.makedirs(save_path, exist_ok=True)
    mi_tensor = mi_tensor.detach().cpu()  # [B*T, H', W']
    
    if mi_tensor.dim() == 3:
        mi_tensor = mi_tensor.unsqueeze(1)  # -> [B*T, 1, H, W]
    elif mi_tensor.dim() != 4:
        raise ValueError(f"Unexpected mi_tensor shape: {mi_tensor.shape}")

    # Resize to [B*T, 1, 224, 224]
    mi_tensor = torch.nn.functional.interpolate(
        mi_tensor, size=(224, 224), mode='bilinear', align_corners=False
    ).squeeze(1)  # -> [B*T, 224, 224]

    for i in range(mi_tensor.shape[0]):
        mi_map = mi_tensor[i]
        mi_map = (mi_map - mi_map.min()) / (mi_map.max() - mi_map.min() + 1e-8)  # Normalize to [0, 1]

        plt.figure(figsize=(2.24, 2.24), dpi=100)  # 224x224 输出
        plt.axis('off')
        plt.imshow(mi_map.numpy(), cmap='jet')  # 你也可以改成 'viridis', 'plasma' 等

        save_file = os.path.join(save_path, f"{video_names[i]}.png")
        plt.savefig(save_file, bbox_inches='tight', pad_inches=0)
        plt.close()

def main():
    # logger
    logger = getLogger(None, __name__)
    dir_name = os.path.splitext(os.path.split(args.cfg)[-1])[0]
    logger.info(f'Load config from {args.cfg}')

    # config
    cfg = Config.fromfile(args.cfg)
    logger.info(cfg.pretty_text)

    # model
    model = build_model(**cfg.model)
    model.load_state_dict(torch.load(args.weights))
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    logger.info('Load trained model %s' % args.weights)

    # Test data
    test_dataset = build_dataset(**cfg.dataset.test)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=cfg.dataset.test.batch_size,
                                                  shuffle=False,
                                                  num_workers=cfg.process.num_works,
                                                  pin_memory=True)
    avg_meter_miou = pyutils.AverageMeter('miou')
    avg_meter_F = pyutils.AverageMeter('F_score')

    # Test
    with torch.no_grad():
        for n_iter, batch_data in enumerate(test_dataloader):
            imgs, audio, mask, video_name_list = batch_data

            imgs = imgs.cuda()
            audio = audio.cuda()
            mask = mask.cuda()
            B, frame, C, H, W = imgs.shape
            imgs = imgs.view(B * frame, C, H, W)
            mask = mask.view(B * frame, H, W)
            audio = audio.view(-1, audio.shape[2],
                               audio.shape[3], audio.shape[4])

            output, _, mi_map = model(audio, imgs)
            if args.save_pred_mask:
                mask_save_path = os.path.join(
                    args.save_dir, dir_name, 'pred_masks')
                save_mask(output.squeeze(1), mask_save_path, video_name_list)
                mi_save_path = os.path.join(args.save_dir, dir_name, 'mi_visuals')
                expanded_video_names = []
                for name in video_name_list:  # 长度 B
                    for f in range(frame):    # 每个视频 T 帧
                        expanded_video_names.append(f"{name}_f{f}")

                save_mi_visualization(mi_map, mi_save_path, expanded_video_names)
            miou = mask_iou(output.squeeze(1), mask)
            avg_meter_miou.add({'miou': miou})
            F_score = Eval_Fmeasure(output.squeeze(1), mask)
            avg_meter_F.add({'F_score': F_score})
            logger.info('n_iter: {}, iou: {}, F_score: {}'.format(
                n_iter, miou, F_score))
        miou = (avg_meter_miou.pop('miou'))
        F_score = (avg_meter_F.pop('F_score'))
        logger.info(f'test miou: {miou.item()}')
        logger.info(f'test F_score: {F_score}')
        logger.info('test miou: {}, F_score: {}'.format(miou.item(), F_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default="/home/songyu/vit-llm/yzy/MI_AVS_Spike/config/ms3/AVSegFormer_pvt2_ms3.py", help='config file path')
    parser.add_argument('--weights', type=str, default="/home/songyu/vit-llm/yzy/MI_AVS_Spike/new_ms3_work_dir/AVSegFormer_pvt2_ms3/ms3_best.pth", help='model weights path')
    parser.add_argument("--save_pred_mask", action='store_true',
                        default=False, help="save predited masks or not")
    parser.add_argument('--save_dir', type=str,
                        default='work_dir', help='save path')
 
    args = parser.parse_args()
    main()
