import os, sys, glob
from pathlib import Path
from typing import List, Dict, Tuple
from functools import partial
import h5py
import numpy as np
import random
import io
from collections import OrderedDict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from argparse import ArgumentParser
import copy

import utils 

## need to wrap tensorflow dataset
import tensorflow as tf
import tensorflow_datasets as tfds

class PreprocFlow(object):

    def __init__(
            self,
            max_value=2**16,
            to_image_coordinates=False,
            to_rgb=False,
            max_speed=1.0
    ):

        self.max_value = max_value
        self.to_image_coordinates = to_image_coordinates
        self.to_tensor = utils.ToTensor(to_float=False)
        if to_rgb:
            self.to_image_coordinates = False
            self.to_tensor = transforms.Compose([
                self.to_tensor, utils.FlowToRgb(max_speed)])

    def __call__(self, arr):

        assert arr.dtype == np.uint16, arr.dtype
        arr = self.to_tensor((arr / self.max_value).astype(np.float32))
        if self.to_image_coordinates:
            arr = torch.stack([-arr[1], arr[0]], 0)
        return arr

class PreprocDepth(object):
    def __init__(
            self,
            max_value=2**16
    ):
        self.max_value = max_value
        self.to_tensor = utils.ToTensor(to_float=False)
    def __call__(self, arr):
        assert arr.dtype == np.uint16, arr.dtype
        arr = self.to_tensor((arr / self.max_value).astype(np.float32))
        return arr

class MoviDataset(Dataset):
    """
    Pytorch Dataset wrapper for sequences of MOVi images.

    Yields dictionary of movie tensors of shape [B,T,C,H,W]
    """

    PASSES_DICT = {
        "images": "video",
        "objects": "segmentations",
        "flow": "forward_flow",
        "depth": "depth",
        "normals": "normal"
    }

    DEFAULT_TRANSFORMS = {
        "images": utils.ToTensor(to_float=False),
        "objects": utils.ToTensor(to_float=False),
        "flow": PreprocFlow(to_image_coordinates=False, to_rgb=False, max_value=1),
        "depth": PreprocDepth(),
        "normals": utils.ToTensor(to_float=False)
    }

    def __init__(
            self,
            dataset_dir: str,
            split: str = 'train',
            sequence_length: int = 2,
            delta_time: int = 1,
            passes: List[str] = ["images", "objects", "flow"],
            passes_dict: Dict[str, str] = PASSES_DICT,
            transform_dict: Dict[str, callable] = DEFAULT_TRANSFORMS,
            resize=None,
            crop_size=None,
            seed: int = 0,
            min_start_frame=0,
            max_start_frame=None,
            shuffle: bool = True,
            to_image=True,
            **kwargs):
        self.dataset_dir = str(dataset_dir)
        self.split = split
        builder = tfds.core.builder_from_directory(self.dataset_dir)
        self.ds = builder.as_dataset(split=self.split, shuffle_files=(split == 'train' and shuffle))
        self.numpy_iterator = iter(tfds.as_numpy(self.ds))

        self.passes = passes
        self.passes_dict = passes_dict
        self.min_start_frame = min_start_frame
        self.max_start_frame = max_start_frame
        self.T = sequence_length
        self.dT = delta_time
        self.to_image = to_image

        self._set_transforms(transform_dict)
        self.init_seed = None

    def _set_transforms(self, Tr_dict):
        self.transform_dict = dict()
        for k in self.passes:
            self.transform_dict[k] = Tr_dict.get(k, utils.ToTensor())

    def __len__(self):
        return len(self.ds)

    def _get_pass(self, data, key='images', frame=0):
        ds_key = self.passes_dict[key]
        return self.transform_dict[key](data[ds_key][frame])

    def _get_pass_movie(self, data, key='images', frame=0, seq_len=1, delta_time=1):
        movie = torch.stack([
            self._get_pass(data, key, frame + t * delta_time)
            for t in range(seq_len)], 0)
        if self.to_image and self.T == 1:
            movie = movie[:,0]
        return movie

    def _init_seed(self):

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

    def reset_iterator(self):
        self.numpy_iterator = iter(tfds.as_numpy(self.ds))

    def __getitem__(self, idx):

        self._init_seed()
        data = next(self.numpy_iterator)
        self.meta = data['metadata']

        ## choose frame
        num_frames = self.meta['num_frames']
        min_frame = min(self.min_start_frame or 0, num_frames - self.T * self.dT - 1)
        max_frame = min((self.max_start_frame or (num_frames - self.T * self.dT)) + self.dT, \
                        num_frames - self.dT)
        self.frame = np.random.randint(min_frame, max_frame)
        return {
            k: self._get_pass_movie(data, key=k, frame=self.frame, seq_len=self.T, delta_time=self.dT)
            for k in self.passes
        }