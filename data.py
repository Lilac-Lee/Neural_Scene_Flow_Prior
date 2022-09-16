#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import numpy as np
from torch.utils.data import Dataset


class FlyingThings3D(Dataset):
    def __init__(self, options, partition='test'):
        self.options = options
        self.partition = partition
        if self.partition == 'train':
            self.datapath = glob.glob(f"{self.options.dataset_path}/TRAIN*.npz")
        else:
            self.datapath = glob.glob(f"{self.options.dataset_path}/TEST*.npz")
        self.cache = {}
        self.cache_size = 30000

        # Bad data..
        bad_data = ["TRAIN_C_0140_left_0006-0",
                    "TRAIN_A_0658_left_0013-0", #pc1 has only 204 valid points (mask).
                    "TRAIN_B_0424_left_0011-0", #pc1 has only 0 valid points.
                    "TRAIN_A_0364_left_0008-0", #pc1 has only 0 valid points.
                    "TRAIN_A_0588_right_0006-0", #pc1 has only 1342 valid points.
                    "TRAIN_A_0074_left_0010-0", #pc1 has only 549 valid points.
                    "TRAIN_B_0727_left_0012-0", #pc1 has only 1456 valid points.
                    "TRAIN_A_0521_right_0012-0",
                    "TRAIN_B_0424_left_0010-0",
                    "TRAIN_A_0074_left_0009-0",
                    "TRAIN_B_0723_right_0006-0",
                    "TRAIN_A_0364_left_0007-0",
                    "TRAIN_B_0424_right_0009-0",
                    "TRAIN_A_0658_left_0012-0",
                    "TRAIN_A_0398_left_0007-0",
                    "TRAIN_A_0433_right_0006-0",
                    "TRAIN_B_0053_left_0009-0",
                    "TRAIN_A_0577_left_0009-0",
                    "TRAIN_A_0074_left_0011-0",
                    "TRAIN_A_0074_left_0011-0",
                    "TRAIN_B_0021_left_0009-0",
                    "TRAIN_B_0727_right_0011-0",
                    "TRAIN_B_0609_right_0009-0",
                    "TRAIN_B_0189_right_0012-0",
                    "TRAIN_B_0189_left_0012-0",
                    "TRAIN_B_0053_left_0011-0",
                    "TRAIN_B_0609_right_0010-0",
                    "TRAIN_B_0609_right_0011-0",
                    "TRAIN_A_0369_right_0009-0",
                    "TRAIN_A_0557_right_0010-0",
                    "TRAIN_A_0047_right_0009-0",
                    "TRAIN_A_0362_right_0008-0",
                    "TRAIN_A_0518_left_0006-0",
                    "TRAIN_A_0074_left_0012-0",
                    "TRAIN_A_0531_right_0006-0",
                    "TRAIN_B_0021_left_0010-0",
                    "TRAIN_B_0189_left_0011-0",
                    "TRAIN_A_0658_left_0014-0",
                    "TRAIN_B_0424_right_0010-0",
                    "TRAIN_A_0369_right_0008-0",
                    "TRAIN_A_0364_left_0009-0",
                    "TEST_A_0149_right_0012-0",
                    "TEST_A_0123_right_0008-0",
                    "TEST_A_0123_right_0009-0",
                    "TEST_A_0149_right_0013-0",
                    "TEST_A_0149_left_0012-0",
                    "TEST_A_0149_left_0011-0",
                    "TEST_A_0023_right_0009-0"]
        self.datapath = [d for d in self.datapath if not any(bad in d for bad in bad_data)]

        print(f"# {self.partition} samples: {len(self.datapath)}")
        
    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        if index in self.cache:
            pc1, pc2, color1, color2, flow = self.cache[index]
        else:
            filename = self.datapath[index]
            with open(filename, 'rb') as fp:
                data = np.load(fp)
                mask1 = data['valid_mask1']
                pc1 = data['points1'].astype('float32')[mask1]
                pc2 = data['points2'].astype('float32')
                color1 = data['color1'].astype('float32')[mask1]
                color2 = data['color2'].astype('float32')
                flow = data['flow'].astype('float32')[mask1]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (pc1, pc2, color1, color2, flow)

        n1 = pc1.shape[0]
        n2 = pc2.shape[0]
        num_points = self.options.num_points
        if self.options.use_all_points:
            num_points = n1

        if n1 >= num_points:
            sample_idx1 = np.random.choice(n1, num_points, replace=False)
        else:
            sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, num_points - n1, replace=True)), axis=-1)

        if n2 >= num_points:
            sample_idx2 = np.random.choice(n2, num_points, replace=False)
        else:
            sample_idx2 = np.concatenate((np.arange(n2), np.random.choice(n2, num_points - n2, replace=True)), axis=-1)

        pc1 = pc1[sample_idx1, :]
        pc2 = pc2[sample_idx2, :]
        flow = flow[sample_idx1, :]

        pc1_center = np.mean(pc1, 0)
        pc1 -= pc1_center
        pc2 -= pc1_center

        return pc1, pc2, flow


class KITTISceneFlowDataset(Dataset):
    def __init__(self, options, train=False):
        self.options = options
        self.train = train
    
        if self.train:
            self.datapath = sorted(glob.glob(f"{self.options.dataset_path}/*.npz"))[:100]
        else:
            self.datapath = sorted(glob.glob(f"{self.options.dataset_path}/*.npz"))[100:]

        self.cache = {}
        self.cache_size = 30000
        
    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, index):
        fn = self.datapath[index]
        with open(fn, 'rb') as fp:
            data = np.load(fp)
            pc1 = data['pos1'].astype('float32')
            pc2 = data['pos2'].astype('float32')
            flow = data['gt'].astype('float32')

        n1 = pc1.shape[0]
        n2 = pc2.shape[0]
        if not self.options.use_all_points:
            num_points = self.options.num_points

            if n1 >= num_points:
                sample_idx1 = np.random.choice(n1, num_points, replace=False)
            else:
                sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, num_points - n1, replace=True)), axis=-1)

            if n2 >= num_points:
                sample_idx2 = np.random.choice(n2, num_points, replace=False)
            else:
                sample_idx2 = np.concatenate((np.arange(n2), np.random.choice(n2, num_points - n2, replace=True)), axis=-1)
                
            pc1 = pc1[sample_idx1, :].astype('float32')
            pc2 = pc2[sample_idx2, :].astype('float32')
            flow = flow[sample_idx1, :].astype('float32')

        return pc1, pc2, flow


class ArgoverseSceneFlowDataset(Dataset):
    def __init__(self, options, partition='val', width=1):
        self.options = options
        self.partition = partition
        self.width = width

        if self.partition == 'train':
            self.datapath = sorted(glob.glob(f"{self.options.dataset_path}/training/*/*/*"))
        elif self.partition == 'test':
            self.datapath = sorted(glob.glob(f"{self.options.dataset_path}/testing/*/*/*"))
        elif self.partition == 'val':
            self.datapath = sorted(glob.glob(f"{self.options.dataset_path}/val/*/*"))
            
    def __len__(self):
        return len(self.datapath)
    
    def __getitem__(self, index):
        filename = self.datapath[index]

        log_id = filename.split('/')[-3]
        dataset_dir = filename.split(log_id)[0]

        with open(filename, 'rb') as fp:
            data = np.load(fp)
            pc1 = data['pc1']
            pc2 = data['pc2']
            flow = data['flow']
            mask1_flow = data['mask1_tracks_flow']
            mask2_flow = data['mask2_tracks_flow']

        n1 = len(pc1)
        n2 = len(pc2)

        full_mask1 = np.arange(n1)
        full_mask2 = np.arange(n2)
        mask1_noflow = np.setdiff1d(full_mask1, mask1_flow, assume_unique=True)
        mask2_noflow = np.setdiff1d(full_mask2, mask2_flow, assume_unique=True)

        if self.options.use_all_points:
            num_points = n1
        else:
            num_points = self.options.num_points
        nonrigid_rate = 0.8
        rigid_rate = 0.2
        if n1 >= num_points:
            if int(num_points * nonrigid_rate) > len(mask1_flow):
                num_points1_flow = len(mask1_flow)
                num_points1_noflow = num_points - num_points1_flow
            else:
                num_points1_flow = int(num_points * nonrigid_rate)
                num_points1_noflow = int(num_points * rigid_rate) + 1

            try:  # ANCHOR: argoverse has some cases without nonrigid flows.
                sample_idx1_noflow = np.random.choice(mask1_noflow, num_points1_noflow, replace=False)
            except:
                sample_idx1_noflow = np.random.choice(mask1_noflow, num_points1_noflow, replace=True)
            sample_idx1_flow = np.random.choice(mask1_flow, num_points1_flow, replace=False)
            sample_idx1 = np.hstack((sample_idx1_flow, sample_idx1_noflow))

            pc1_ = pc1[sample_idx1, :]
            flow_ = flow[sample_idx1, :]

            pc1 = pc1_.astype('float32')
            flow = flow_.astype('float32')

        if n2 >= num_points:
            if int(num_points * nonrigid_rate) > len(mask2_flow):
                num_points2_flow = len(mask2_flow)
                num_points2_noflow = num_points - num_points2_flow
            else:
                num_points2_flow = int(num_points * nonrigid_rate)
                num_points2_noflow = int(num_points * rigid_rate) + 1
                
            try:  # ANCHOR: argoverse has some cases without nonrigid flows.
                sample_idx2_noflow = np.random.choice(mask2_noflow, num_points2_noflow, replace=False)
            except:
                sample_idx2_noflow = np.random.choice(mask2_noflow, num_points2_noflow, replace=True)
            sample_idx2_flow = np.random.choice(mask2_flow, num_points2_flow, replace=False)
            sample_idx2 = np.hstack((sample_idx2_flow, sample_idx2_noflow))

            pc2_ = pc2[sample_idx2, :]
            pc2 = pc2_.astype('float32')

        return pc1, pc2, flow


class NuScenesSceneFlowDataset(Dataset):
    def __init__(self, options, partition="val", width=1):
        self.options = options
        self.partition = partition
        self.width = width

        if self.partition == "train":
            self.datapath = sorted(glob.glob(f"{self.options.dataset_path}/train/*"))
        elif self.partition == "val":
            self.datapath = sorted(glob.glob(f"{self.options.dataset_path}/val/*"))
            
        # Bad data. Pretty noisy samples.
        bad_data = ["d02c6908713147e9a4ac5d50784815d3",
                    "ae989fac82d248b98ce769e753f60f87",
                    "365e72358ddb405e953cdad865815966",
                    "4a16cf07faf54dbf93e0c4c083b38c63",
                    "44fd8959bd574d7fb6773a9fe341282e",
                    "c6879ea1c3d845eebd7825e6e454bee1",
                    "359023a812c24fbcae41334842672dd2",
                    "aa5d89b9f988450eaa442070576913b7",
                    "c4344682d52f4578b5aa983612764e9b",
                    "6fd5607c93fa4b569eb2bd0d7f30f9a0",
                    "5ab1e1f0829541269856edca0f7517da",
                    "1c01cb36784e44fc8a5ef7d9689ef2fd",
                    "15a106fd45604b6bb85d67c1e5033022",
                    "6803c2feca4b40e78434cf209ee8c2da",
                    "6737346ecd5144d28cef656f17953959",
        ]
        self.datapath = [d for d in self.datapath if not any(bad in d for bad in bad_data)]

    def __getitem__(self, index):
        filename = self.datapath[index]

        with open(filename, 'rb') as fp:
            data = np.load(fp)
            pc1 = data['pc1'].astype('float32')
            pc2 = data['pc2'].astype('float32')
            flow = data['flow'].astype('float32')
            mask1_flow = data['mask1_tracks_flow']
            mask2_flow = data['mask2_tracks_flow']

        n1 = len(pc1)
        n2 = len(pc2)

        full_mask1 = np.arange(n1)
        full_mask2 = np.arange(n2)
        mask1_noflow = np.setdiff1d(full_mask1, mask1_flow, assume_unique=True)
        mask2_noflow = np.setdiff1d(full_mask2, mask2_flow, assume_unique=True)

        if self.options.use_all_points:
            num_points = n1
        else:
            num_points = self.options.num_points
            nonrigid_rate = 0.8
            rigid_rate = 0.2
            if n1 >= num_points:
                if int(num_points * nonrigid_rate) > len(mask1_flow):
                    num_points1_flow = len(mask1_flow)
                    num_points1_noflow = num_points - num_points1_flow
                else:
                    num_points1_flow = int(num_points * nonrigid_rate)
                    num_points1_noflow = int(num_points * rigid_rate) + 1
                sample_idx1_flow = np.random.choice(mask1_flow, num_points1_flow, replace=False)
                try:  # ANCHOR: nuscenes has some cases without nonrigid flows.
                    sample_idx1_noflow = np.random.choice(mask1_noflow, num_points1_noflow, replace=False)
                except:
                    sample_idx1_noflow = np.random.choice(mask1_noflow, num_points1_noflow, replace=True)
                sample_idx1 = np.hstack((sample_idx1_flow, sample_idx1_noflow))

            else:
                sample_idx1 = np.concatenate((np.arange(n1), np.random.choice(n1, num_points - n1, replace=True)), axis=-1)
            pc1_ = pc1[sample_idx1, :]
            flow_ = flow[sample_idx1, :]

            pc1 = pc1_.astype('float32')
            flow = flow_.astype('float32')

            if n2 >= num_points:
                if int(num_points * nonrigid_rate) > len(mask2_flow):
                    num_points2_flow = len(mask2_flow)
                    num_points2_noflow = num_points - num_points2_flow
                else:
                    num_points2_flow = int(num_points * nonrigid_rate)
                    num_points2_noflow = int(num_points * rigid_rate) + 1
                sample_idx2_flow = np.random.choice(mask2_flow, num_points2_flow, replace=False)
                sample_idx2_noflow = np.random.choice(mask2_noflow, num_points2_noflow, replace=False)
                sample_idx2 = np.hstack((sample_idx2_flow, sample_idx2_noflow))
            else:
                sample_idx2 = np.concatenate((np.arange(n2), np.random.choice(n2, num_points - n2, replace=True)), axis=-1)

            pc2_ = pc2[sample_idx2, :]
            pc2 = pc2_.astype('float32')

        return pc1, pc2, flow

    def __len__(self):
        return len(self.datapath)

