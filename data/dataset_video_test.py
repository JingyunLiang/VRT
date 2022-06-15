# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import os
import glob
import torch
from os import path as osp
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

import utils.utils_video as utils_video


class VideoRecurrentTestDataset(data.Dataset):
    """Video test dataset for recurrent architectures, which takes LR video
    frames as input and output corresponding HR video frames. Modified from
    https://github.com/xinntao/BasicSR/blob/master/basicsr/data/reds_dataset.py

    Supported datasets: Vid4, REDS4, REDSofficial.
    More generally, it supports testing dataset with following structures:

    dataroot
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── ...

    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            cache_data (bool): Whether to cache testing datasets.
            name (str): Dataset name.
            meta_info_file (str): The path to the file storing the list of test
                folders. If not provided, all the folders in the dataroot will
                be used.
            num_frame (int): Window size for input frames.
            padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(VideoRecurrentTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}

        self.imgs_lq, self.imgs_gt = {}, {}
        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
                subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))
            subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, '*')))

        for subfolder_lq, subfolder_gt in zip(subfolders_lq, subfolders_gt):
            # get frame list for lq and gt
            subfolder_name = osp.basename(subfolder_lq)
            img_paths_lq = sorted(list(utils_video.scandir(subfolder_lq, full_path=True)))
            img_paths_gt = sorted(list(utils_video.scandir(subfolder_gt, full_path=True)))

            max_idx = len(img_paths_lq)
            assert max_idx == len(img_paths_gt), (f'Different number of images in lq ({max_idx})'
                                                  f' and gt folders ({len(img_paths_gt)})')

            self.data_info['lq_path'].extend(img_paths_lq)
            self.data_info['gt_path'].extend(img_paths_gt)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append(f'{i}/{max_idx}')
            border_l = [0] * max_idx
            for i in range(self.opt['num_frame'] // 2):
                border_l[i] = 1
                border_l[max_idx - i - 1] = 1
            self.data_info['border'].extend(border_l)

            # cache data or save the frame list
            if self.cache_data:
                print(f'Cache {subfolder_name} for VideoTestDataset...')
                self.imgs_lq[subfolder_name] = utils_video.read_img_seq(img_paths_lq)
                self.imgs_gt[subfolder_name] = utils_video.read_img_seq(img_paths_gt)
            else:
                self.imgs_lq[subfolder_name] = img_paths_lq
                self.imgs_gt[subfolder_name] = img_paths_gt

        # Find unique folder strings
        self.folders = sorted(list(set(self.data_info['folder'])))
        self.sigma = opt['sigma'] / 255. if 'sigma' in opt else 0 # for non-blind video denoising

    def __getitem__(self, index):
        folder = self.folders[index]

        if self.sigma:
        # for non-blind video denoising
            if self.cache_data:
                imgs_gt = self.imgs_gt[folder]
            else:
                imgs_gt = utils_video.read_img_seq(self.imgs_gt[folder])

            torch.manual_seed(0)
            noise_level = torch.ones((1, 1, 1, 1)) * self.sigma
            noise = torch.normal(mean=0, std=noise_level.expand_as(imgs_gt))
            imgs_lq = imgs_gt + noise
            t, _, h, w = imgs_lq.shape
            imgs_lq = torch.cat([imgs_lq, noise_level.expand(t, 1, h, w)], 1)
        else:
        # for video sr and deblurring
            if self.cache_data:
                imgs_lq = self.imgs_lq[folder]
                imgs_gt = self.imgs_gt[folder]
            else:
                imgs_lq = utils_video.read_img_seq(self.imgs_lq[folder])
                imgs_gt = utils_video.read_img_seq(self.imgs_gt[folder])

        return {
            'L': imgs_lq,
            'H': imgs_gt,
            'folder': folder,
            'lq_path': self.imgs_lq[folder],
        }

    def __len__(self):
        return len(self.folders)


class SingleVideoRecurrentTestDataset(data.Dataset):
    """Single video test dataset for recurrent architectures, which takes LR video
    frames as input and output corresponding HR video frames (only input LQ path).

    More generally, it supports testing dataset with following structures:

    dataroot
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── ...

    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            cache_data (bool): Whether to cache testing datasets.
            name (str): Dataset name.
            meta_info_file (str): The path to the file storing the list of test
                folders. If not provided, all the folders in the dataroot will
                be used.
            num_frame (int): Window size for input frames.
            padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(SingleVideoRecurrentTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.lq_root = opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'folder': [], 'idx': [], 'border': []}

        self.imgs_lq = {}
        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))

        for subfolder_lq in subfolders_lq:
            # get frame list for lq and gt
            subfolder_name = osp.basename(subfolder_lq)
            img_paths_lq = sorted(list(utils_video.scandir(subfolder_lq, full_path=True)))

            max_idx = len(img_paths_lq)

            self.data_info['lq_path'].extend(img_paths_lq)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            for i in range(max_idx):
                self.data_info['idx'].append(f'{i}/{max_idx}')
            border_l = [0] * max_idx
            for i in range(self.opt['num_frame'] // 2):
                border_l[i] = 1
                border_l[max_idx - i - 1] = 1
            self.data_info['border'].extend(border_l)

            # cache data or save the frame list
            if self.cache_data:
                print(f'Cache {subfolder_name} for VideoTestDataset...')
                self.imgs_lq[subfolder_name] = utils_video.read_img_seq(img_paths_lq)
            else:
                self.imgs_lq[subfolder_name] = img_paths_lq

        # Find unique folder strings
        self.folders = sorted(list(set(self.data_info['folder'])))

    def __getitem__(self, index):
        folder = self.folders[index]

        if self.cache_data:
            imgs_lq = self.imgs_lq[folder]
        else:
            imgs_lq = utils_video.read_img_seq(self.imgs_lq[folder])

        return {
            'L': imgs_lq,
            'folder': folder,
            'lq_path': self.imgs_lq[folder],
        }

    def __len__(self):
        return len(self.folders)


class VideoTestVimeo90KDataset(data.Dataset):
    """Video test dataset for Vimeo90k-Test dataset.

    It only keeps the center frame for testing.
    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            cache_data (bool): Whether to cache testing datasets.
            name (str): Dataset name.
            meta_info_file (str): The path to the file storing the list of test
                folders. If not provided, all the folders in the dataroot will
                be used.
            num_frame (int): Window size for input frames.
            padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(VideoTestVimeo90KDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        temporal_scale = opt.get('temporal_scale', 1)
        if self.cache_data:
            raise NotImplementedError('cache_data in Vimeo90K-Test dataset is not implemented.')
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}
        neighbor_list = [i + (9 - opt['num_frame']) // 2 for i in range(opt['num_frame'])][:: temporal_scale]

        with open(opt['meta_info_file'], 'r') as fin:
            subfolders = [line.split(' ')[0] for line in fin]
        for idx, subfolder in enumerate(subfolders):
            gt_path = osp.join(self.gt_root, subfolder, 'im4.png')
            self.data_info['gt_path'].append(gt_path)
            lq_paths = [osp.join(self.lq_root, subfolder, f'im{i}.png') for i in neighbor_list]
            self.data_info['lq_path'].append(lq_paths)
            self.data_info['folder'].append(subfolder)
            self.data_info['idx'].append(f'{idx}/{len(subfolders)}')
            self.data_info['border'].append(0)

        self.pad_sequence = opt.get('pad_sequence', False)
        self.mirror_sequence = opt.get('mirror_sequence', False)

    def __getitem__(self, index):
        lq_path = self.data_info['lq_path'][index]
        gt_path = self.data_info['gt_path'][index]
        imgs_lq = utils_video.read_img_seq(lq_path)
        img_gt = utils_video.read_img_seq([gt_path])

        if self.pad_sequence:  # pad the sequence: 7 frames to 8 frames
            imgs_lq = torch.cat([imgs_lq, imgs_lq[-1:,...]], dim=0)

        if self.mirror_sequence:  # mirror the sequence: 7 frames to 14 frames
            imgs_lq = torch.cat([imgs_lq, imgs_lq.flip(0)], dim=0)

        return {
            'L': imgs_lq,  # (t, c, h, w)
            'H': img_gt,  # (c, h, w)
            'folder': self.data_info['folder'][index],  # folder name
            'idx': self.data_info['idx'][index],  # e.g., 0/843
            'border': self.data_info['border'][index],  # 0 for non-border
            'lq_path': lq_path,
            'gt_path': [gt_path]
        }

    def __len__(self):
        return len(self.data_info['gt_path'])


class VFI_DAVIS(data.Dataset):
    """Video test dataset for DAVIS dataset in video frame interpolation.
    Modified from https://github.com/tarun005/FLAVR/blob/main/dataset/Davis_test.py
    """

    def __init__(self, data_root, ext="png"):

        super().__init__()

        self.data_root = data_root
        self.images_sets = []

        for label_id in os.listdir(self.data_root):
            ctg_imgs_ = sorted(os.listdir(os.path.join(self.data_root , label_id)))
            ctg_imgs_ = [os.path.join(self.data_root , label_id , img_id) for img_id in ctg_imgs_]
            for start_idx in range(0,len(ctg_imgs_)-6,2):
                add_files = ctg_imgs_[start_idx : start_idx+7 : 2]
                add_files = add_files[:2] + [ctg_imgs_[start_idx+3]] + add_files[2:]
                self.images_sets.append(add_files)

        self.transforms = transforms.Compose([
                transforms.CenterCrop((480, 840)),
                transforms.ToTensor()
            ])

    def __getitem__(self, idx):

        imgpaths = self.images_sets[idx]
        images = [Image.open(img) for img in imgpaths]
        images = [self.transforms(img) for img in images]

        return {
            'L': torch.stack(images[:2] + images[3:], 0),
            'H': images[2].unsqueeze(0),
            'folder': str(idx),
            'gt_path': ['vfi_result.png'],
        }

    def __len__(self):
        return len(self.images_sets)


class VFI_UCF101(data.Dataset):
    """Video test dataset for UCF101 dataset in video frame interpolation.
        Modified from https://github.com/tarun005/FLAVR/blob/main/dataset/ucf101_test.py
    """

    def __init__(self, data_root, ext="png"):
        super().__init__()

        self.data_root = data_root
        self.file_list = sorted(os.listdir(self.data_root))

        self.transforms = transforms.Compose([
                transforms.CenterCrop((224,224)),
                transforms.ToTensor(),
            ])

    def __getitem__(self, idx):

        imgpath = os.path.join(self.data_root , self.file_list[idx])
        imgpaths = [os.path.join(imgpath , "frame0.png") , os.path.join(imgpath , "frame1.png") ,os.path.join(imgpath , "frame2.png") ,os.path.join(imgpath , "frame3.png") ,os.path.join(imgpath , "framet.png")]

        images = [Image.open(img) for img in imgpaths]
        images = [self.transforms(img) for img in images]

        return {
            'L': torch.stack(images[:-1], 0),
            'H': images[-1].unsqueeze(0),
            'folder': self.file_list[idx],
            'gt_path': ['vfi_result.png'],
        }

    def __len__(self):
        return len(self.file_list)


class VFI_Vid4(data.Dataset):
    """Video test dataset for Vid4 dataset in video frame interpolation.
    Modified from https://github.com/tarun005/FLAVR/blob/main/dataset/Davis_test.py
    """

    def __init__(self, data_root, ext="png"):

        super().__init__()

        self.data_root = data_root
        self.images_sets = []
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': []}
        self.lq_path = []
        self.folder = []

        for label_id in os.listdir(self.data_root):
            ctg_imgs_ = sorted(os.listdir(os.path.join(self.data_root, label_id)))
            ctg_imgs_ = [os.path.join(self.data_root , label_id , img_id) for img_id in ctg_imgs_]
            if len(ctg_imgs_) % 2 == 0:
                ctg_imgs_.append(ctg_imgs_[-1])
            ctg_imgs_.insert(0, None)
            ctg_imgs_.insert(0, ctg_imgs_[1])
            ctg_imgs_.append(None)
            ctg_imgs_.append(ctg_imgs_[-2])

            for start_idx in range(0,len(ctg_imgs_)-6,2):
                add_files = ctg_imgs_[start_idx : start_idx+7 : 2]
                self.data_info['lq_path'].append([os.path.basename(path) for path in add_files])
                self.data_info['gt_path'].append(os.path.basename(ctg_imgs_[start_idx + 3]))
                self.data_info['folder'].append(label_id)
                add_files = add_files[:2] + [ctg_imgs_[start_idx+3]] + add_files[2:]
                self.images_sets.append(add_files)

        self.transforms = transforms.Compose([
                transforms.ToTensor()
            ])

    def __getitem__(self, idx):
        imgpaths = self.images_sets[idx]
        images = [Image.open(img) for img in imgpaths]
        images = [self.transforms(img) for img in images]

        return {
            'L': torch.stack(images[:2] + images[3:], 0),
            'H': images[2].unsqueeze(0),
            'folder': self.data_info['folder'][idx],
            'lq_path': self.data_info['lq_path'][idx],
            'gt_path': [self.data_info['gt_path'][idx]]
        }

    def __len__(self):
        return len(self.images_sets)
