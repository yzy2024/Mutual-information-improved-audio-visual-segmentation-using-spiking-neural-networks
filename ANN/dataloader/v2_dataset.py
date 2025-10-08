import os
# from wave import _wave_params
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd
import pickle
import json

# import cv2
from PIL import Image
from torchvision import transforms

# from .config import cfg_avs

import librosa


def wavfile_to_examples(wav_file, target_sample_rate=16000):
    """
    Load a .wav file and convert to VGGish-compatible [N, 1, 96, 64] Tensor
    N is the number of time windows (frames of 0.96s each)

    Parameters:
        wav_file (str): path to .wav file
        target_sample_rate (int): 16000 Hz expected by VGGish

    Returns:
        np.ndarray: shape [N, 1, 96, 64], dtype float32
    """
    # Step 1: Load audio
    waveform, sr = librosa.load(wav_file, sr=target_sample_rate)  # [T,] waveform
    
    # Step 2: Extract log-mel spectrogram
    # Each frame: 25ms window with 10ms hop
    n_fft = int(0.025 * sr)         # 400
    hop_length = int(0.010 * sr)    # 160
    n_mels = 64

    mel_spec = librosa.feature.melspectrogram(
        y=waveform, 
        sr=sr, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        n_mels=n_mels,
        power=2.0  # power spectrogram (not amplitude)
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  # shape: [64, T]

    # Step 3: Frame into 0.96s examples (i.e., 96 frames @ 10ms hop = 0.96s)
    num_frames = log_mel_spec.shape[1]
    window_size = 96
    if num_frames < window_size:
        # Pad with zeros
        pad_width = window_size - num_frames
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        num_frames = window_size
    
    # Now split into segments of 96 frames
    examples = []
    for i in range(0, num_frames - window_size + 1, window_size):
        example = log_mel_spec[:, i:i + window_size]
        examples.append(example)

    examples = np.stack(examples, axis=0)  # [N, 64, 96]
    examples = examples[:, np.newaxis, :, :]  # [N, 1, 64, 96]
    examples = examples.astype(np.float32)
    return examples

def get_v2_pallete(label_to_idx_path, num_cls=71):
    def _getpallete(num_cls=71):
        """build the unified color pallete for AVSBench-object (V1) and AVSBench-semantic (V2),
        71 is the total category number of V2 dataset, you should not change that"""
        n = num_cls
        pallete = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            pallete[j * 3 + 0] = 0
            pallete[j * 3 + 1] = 0
            pallete[j * 3 + 2] = 0
            i = 0
            while (lab > 0):
                pallete[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                pallete[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                pallete[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i = i + 1
                lab >>= 3
        return pallete  # list, lenth is n_classes*3

    with open(label_to_idx_path, 'r') as fr:
        label_to_pallete_idx = json.load(fr)
    v2_pallete = _getpallete(num_cls)  # list
    v2_pallete = np.array(v2_pallete).reshape(-1, 3)
    assert len(v2_pallete) == len(label_to_pallete_idx)
    return v2_pallete


def crop_resize_img(crop_size, img, img_is_mask=False):
    outsize = crop_size
    short_size = outsize
    w, h = img.size
    if w > h:
        oh = short_size
        ow = int(1.0 * w * oh / h)
    else:
        ow = short_size
        oh = int(1.0 * h * ow / w)
    if not img_is_mask:
        img = img.resize((ow, oh), Image.BILINEAR)
    else:
        img = img.resize((ow, oh), Image.NEAREST)
    # center crop
    w, h = img.size
    x1 = int(round((w - outsize) / 2.))
    y1 = int(round((h - outsize) / 2.))
    img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
    # print("crop for train. set")
    return img


def resize_img(crop_size, img, img_is_mask=False):
    outsize = crop_size
    # only resize for val./test. set
    if not img_is_mask:
        img = img.resize((outsize, outsize), Image.BILINEAR)
    else:
        img = img.resize((outsize, outsize), Image.NEAREST)
    return img


def color_mask_to_label(mask, v_pallete):
    mask_array = np.array(mask).astype('int32')
    semantic_map = []
    for colour in v_pallete:
        equality = np.equal(mask_array, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    # pdb.set_trace() # there is only one '1' value for each pixel, run np.sum(semantic_map, axis=-1)
    label = np.argmax(semantic_map, axis=-1)
    return label


def load_image_in_PIL_to_Tensor(path, split='train', mode='RGB', transform=None, cfg=None):
    img_PIL = Image.open(path).convert(mode)
    if cfg.crop_img_and_mask:
        if split == 'train':
            img_PIL = crop_resize_img(
                cfg.crop_size, img_PIL, img_is_mask=False)
        else:
            img_PIL = resize_img(cfg.crop_size,
                                 img_PIL, img_is_mask=False)
    if transform:
        img_tensor = transform(img_PIL)
        return img_tensor
    return img_PIL


def load_color_mask_in_PIL_to_Tensor(path, v_pallete, split='train', mode='RGB', cfg=None):
    color_mask_PIL = Image.open(path).convert(mode)
    if cfg.crop_img_and_mask:
        if split == 'train':
            color_mask_PIL = crop_resize_img(
                cfg.crop_size, color_mask_PIL, img_is_mask=True)
        else:
            color_mask_PIL = resize_img(
                cfg.crop_size, color_mask_PIL, img_is_mask=True)
    # obtain semantic label
    color_label = color_mask_to_label(color_mask_PIL, v_pallete)
    color_label = torch.from_numpy(color_label)  # [H, W]
    color_label = color_label.unsqueeze(0)
    # binary_mask = (color_label != (cfg_avs.NUM_CLASSES-1)).float()
    # return color_label, binary_mask # both [1, H, W]
    return color_label  # both [1, H, W]


def load_audio_lm(audio_lm_path):
    with open(audio_lm_path, 'rb') as fr:
        audio_log_mel = pickle.load(fr)
    audio_log_mel = audio_log_mel.detach()  # [5, 1, 96, 64]
    return audio_log_mel


class V2Dataset(Dataset):
    """Dataset for audio visual semantic segmentation of AVSBench-semantic (V2)"""

    def __init__(self, split='train', cfg=None, debug_flag=False):
        super(V2Dataset, self).__init__()
        self.split = split
        self.cfg = cfg
        self.mask_num = cfg.mask_num
        df_all = pd.read_csv(cfg.meta_csv_path, sep=',')
        self.df_split = df_all[df_all['split'] == split]
        if debug_flag:
            self.df_split = self.df_split[:100]
        print("{}/{} videos are used for {}.".format(len(self.df_split),
              len(df_all), self.split))
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        self.v2_pallete = get_v2_pallete(
            cfg.label_idx_path, num_cls=cfg.num_class)
        valid_indices = []
        # for i in range(len(self.df_split)):
        #     row = self.df_split.iloc[i]
        #     video_name, set = row['uid'], row['label']
        #     path = os.path.join(self.cfg.dir_base, set, video_name, 'frames')
        #     if os.path.exists(path):
        #         valid_indices.append(i)
        # self.df_split = self.df_split.iloc[valid_indices].reset_index(drop=True)


    def __getitem__(self, index):
        df_one_video = self.df_split.iloc[index]
        video_name, set = df_one_video['uid'], df_one_video['label']
        img_base_path = os.path.join(
            self.cfg.dir_base, set, video_name, 'frames')
        audio_path = os.path.join(
            self.cfg.dir_base, set, video_name, 'audio.wav')
        color_mask_base_path = os.path.join(
            self.cfg.dir_base, set, video_name, 'labels_rgb')

        # data from AVSBench-object single-source subset (5s, gt is only the first annotated frame)
        if set == 'v1s':
            vid_temporal_mask_flag = torch.Tensor(
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])  # .bool()
            gt_temporal_mask_flag = torch.Tensor(
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # .bool()
        # data from AVSBench-object multi-sources subset (5s, all 5 extracted frames are annotated)
        elif set == 'v1m':
            vid_temporal_mask_flag = torch.Tensor(
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])  # .bool()
            gt_temporal_mask_flag = torch.Tensor(
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])  # .bool()
        # data from newly collected videos in AVSBench-semantic (10s, all 10 extracted frames are annotated))
        elif set == 'v2':
            vid_temporal_mask_flag = torch.ones(10)  # .bool()
            gt_temporal_mask_flag = torch.ones(10)  # .bool()

        img_path_list = sorted(os.listdir(img_base_path)
                               )  # 5 for v1, 10 for new v2
        imgs_num = len(img_path_list)
        imgs_pad_zero_num = 10 - imgs_num
        imgs = []
        for img_id in range(imgs_num):
            img_path = os.path.join(img_base_path, "%d.jpg" % (img_id))
            img = load_image_in_PIL_to_Tensor(
                img_path, split=self.split, transform=self.img_transform, cfg=self.cfg)
            imgs.append(img)
        for pad_i in range(imgs_pad_zero_num):  # ! pad black image?
            img = torch.zeros_like(img)
            imgs.append(img)

        labels = []
        mask_path_list = sorted(os.listdir(color_mask_base_path))
        for mask_path in mask_path_list:
            if not mask_path.endswith(".png"):
                mask_path_list.remove(mask_path)
        mask_num = len(mask_path_list)
        if self.split != 'train':
            if set == 'v2':
                assert mask_num == 10
            else:
                assert mask_num == 5

        mask_num = len(mask_path_list)
        label_pad_zero_num = 10 - mask_num
        for mask_id in range(mask_num):
            mask_path = os.path.join(
                color_mask_base_path, "%d.png" % (mask_id))
            # mask_path =  os.path.join(color_mask_base_path, mask_path_list[mask_id])
            color_label = load_color_mask_in_PIL_to_Tensor(
                mask_path, v_pallete=self.v2_pallete, split=self.split, cfg=self.cfg)
            # print('color_label.shape: ', color_label.shape)
            labels.append(color_label)
        for pad_j in range(label_pad_zero_num):
            color_label = torch.zeros_like(color_label)
            labels.append(color_label)

        imgs_tensor = torch.stack(imgs, dim=0)
        labels_tensor = torch.stack(labels, dim=0)
        # audio_path = load_audio_lm(audio_path)
        # audio_tensor = wavfile_to_examples(audio_path)  # [N, 1, 96, 64]，N=音频窗口数
        # audio_tensor = torch.from_numpy(audio_tensor).float()  # 转为 Tensor
        # audio_tensor = audio_tensor.transpose(2, 3)
        # max_audio_len = 5
        # audio_len = audio_tensor.shape[0]

        # if audio_len < max_audio_len:
        #     # pad with zeros
        #     pad_shape = (max_audio_len - audio_len, 1, 96, 64)
        #     pad = torch.zeros(pad_shape, dtype=audio_tensor.dtype)
        #     audio_tensor = torch.cat([audio_tensor, pad], dim=0)
        # if audio_len > max_audio_len:
        #     # crop
        #     audio_tensor = audio_tensor[:max_audio_len]
        # audio_tensor = audio_tensor.reshape(max_audio_len, 1, 96, 64)

        return imgs_tensor, audio_path, labels_tensor, \
            vid_temporal_mask_flag, gt_temporal_mask_flag, video_name
        # return imgs_tensor, audio_tensor, labels_tensor, \
        #     vid_temporal_mask_flag, gt_temporal_mask_flag, video_name

    def __len__(self):
        return len(self.df_split)

    @property
    def num_classes(self):
        """Number of categories (including background)."""
        return self.cfg.num_class

    @property
    def classes(self):
        """Category names."""
        with open(self.cfg.label_idx_path, 'r') as fr:
            classes = json.load(fr)
        return [label for label in classes.keys()]
