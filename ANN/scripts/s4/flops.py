import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table

cfg_path = '/home/songyu/vit-llm/yzy/MI_AVS/config/s4/AVSegFormer_pvt2_s4.py'
weights_path = '/home/songyu/vit-llm/yzy/MI_AVS/work_dir/AVSegFormer_pvt2_s4/s4_best.pth'

import torch.nn
import os
from mmcv import Config
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils import pyutils
from utility import mask_iou, Eval_Fmeasure, save_mask
from utils.logger import getLogger
from model import build_model
from dataloader import build_dataset


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
        # [bs, 5, 3, 224, 224], [bs, 5, 1, 96, 64], [bs, 1, 1, 224, 224]
        imgs, audio, mask, category_list, video_name_list = batch_data

        imgs = imgs.cuda()
        audio = audio.cuda()
        mask = mask.cuda()
        B, frame, C, H, W = imgs.shape
        imgs = imgs.view(B * frame, C, H, W)
        mask = mask.view(B * frame, H, W)
        audio = audio.view(-1, audio.shape[2],
                            audio.shape[3], audio.shape[4])

        output, _, _ = model(audio, imgs)
        if args.save_pred_mask:
            mask_save_path = os.path.join(
                args.save_dir, dir_name, 'pred_masks')
            save_mask(output.squeeze(1), mask_save_path,
                        category_list, video_name_list)

        miou = mask_iou(output.squeeze(1), mask)
        avg_meter_miou.add({'miou': miou})
        F_score = Eval_Fmeasure(output.squeeze(1), mask)
        avg_meter_F.add({'F_score': F_score})
        logger.info('n_iter: {}, iou: {}, F_score: {}'.format(
            n_iter, miou, F_score))
        flops = FlopCountAnalysis(model, (dummy_audio, dummy_img))
        print("Total FLOPs: {:.2f} GFLOPs".format(flops.total() / 1e9))
        print(parameter_count_table(model))
        break
    
    miou = (avg_meter_miou.pop('miou'))
    F_score = (avg_meter_F.pop('F_score'))
    logger.info(f'test miou: {miou.item}')
    logger.info(f'test F_score: {F_score}')
    logger.info('test miou: {}, F_score: {}'.format(miou.item(), F_score))
# 计算 FLOPs

