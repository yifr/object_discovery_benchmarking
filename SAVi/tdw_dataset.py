import os
import json
import glob
import pdb
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt



class TDWDataset(Dataset):
    def __init__(self, dataset_dir, training, flow_threshold=0.5, delta_time=1, frame_idx=5,
                 target="rgb_flow"):
        self.training = training
        self.frame_idx = frame_idx
        self.delta_time = delta_time
        self.flow_threshold = flow_threshold
        self.target = target
        assert self.flow_threshold == 0.5, "The threshold for RAFT flow should be 0.5"

        meta_path = os.path.join(dataset_dir, 'meta.json')
        self.meta = json.loads(Path(meta_path).open().read())

        if self.training:
            self.file_list = glob.glob(os.path.join(dataset_dir, 'images', 'model_split_[0-9]*', '*[0-8]'))
        else:
            self.file_list = glob.glob(os.path.join(dataset_dir, 'images', 'model_split_[0-3]', '*9'))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        while True:
            try:
                file_name = self.file_list[idx]
                image = self.read_frame(file_name, frame_idx=self.frame_idx)
                # image_2 = self.read_frame(file_name, frame_idx=self.frame_idx+self.delta_time)
                raft_moving = self.prepare_motion_segments(file_name)
                segment_colors = self.read_frame(file_name.replace('/images/', '/objects/'), frame_idx=self.frame_idx)
                _, segment_map, gt_moving = self.process_segmentation_color(segment_colors, file_name)

                return image, segment_map, gt_moving, raft_moving

            except Exception as e:
                idx = idx + 1
                print(e)
                print('Loading from %s instead' % self.file_list[idx])

    @staticmethod
    def read_frame(path, frame_idx):
        image_path = os.path.join(path, format(frame_idx, '05d') + '.png')
        return read_image(image_path)

    @staticmethod
    def _object_id_hash(objects, val=256, dtype=torch.long):
        C = objects.shape[0]
        objects = objects.to(dtype)
        out = torch.zeros_like(objects[0:1, ...])
        for c in range(C):
            scale = val ** (C - 1 - c)
            out += scale * objects[c:c + 1, ...]
        return out

    def process_segmentation_color(self, seg_color, file_name):
        # convert segmentation color to integer segment id
        raw_segment_map = self._object_id_hash(seg_color, val=256, dtype=torch.long)
        raw_segment_map = raw_segment_map.squeeze(0)

        # remove zone id from the raw_segment_map
        meta_key = 'playroom_large_v3_images/' + file_name.split('/images/')[-1] + '.hdf5'
        zone_id = int(self.meta[meta_key]['zone'])
        raw_segment_map[raw_segment_map == zone_id] = 0

        # convert raw segment ids to a range in [0, n]
        _, segment_map = torch.unique(raw_segment_map, return_inverse=True)
        segment_map -= segment_map.min()

        # gt_moving_mask
        gt_moving = raw_segment_map == int(self.meta[meta_key]['moving'])

        return raw_segment_map, segment_map, gt_moving

    @staticmethod
    def flow_to_segment(flow, thresh):
        magnitude = (flow ** 2).sum(1) ** 0.5
        motion_segment = (magnitude > thresh).unsqueeze(1)
        return magnitude, motion_segment

    def prepare_motion_segments(self, file_name):
        load_path = file_name.replace('/images/', '/flows/') + '.pt'
        raft_flow = torch.load(load_path)
        _, motion_segment = self.flow_to_segment(raft_flow[None], thresh=self.flow_threshold)
        return motion_segment[0, 0]


if __name__ == "__main__":
    dataset_dir = '/om2/user/yyf/tdw_playroom_small'
    batch_size = 10
    flow_threshold = 0.5

    train_dataset = TDWDataset(dataset_dir, flow_threshold=flow_threshold, training=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TDWDataset(dataset_dir, flow_threshold=flow_threshold, training=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    image, segment_map, gt_moving, raft_moving = next(iter(val_dataloader))

    print(raft_moving.any())
    print((raft_moving == True).any())
    print(np.where(raft_moving))
