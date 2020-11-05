import torch
from torchvision import datasets, transforms

import sys, os
import numpy as np
import h5py
from matplotlib import pyplot as plt
import glob
from pathlib import Path
import time
from functools import partial
from multiprocessing import Pool
import pickle
import datetime as dt
import json

sys.path.append(str(Path(os.getcwd()).parent))
sys.path.append(os.path.join(os.getcwd(), 'ye'))

def subsample(x, n, m):
    return x[..., n[0]:n[1], m[0]:m[1]]

def load_h5_file(file_path):
    """
    Given a file path to an h5 file assumed to house a tensor, 
    load that tensor into memory and return a pointer.
    """
    # load
    fr = h5py.File(file_path, 'r')
    a_group_key = list(fr.keys())[0]
    data = list(fr[a_group_key])
    # transform to appropriate numpy array 
    data=data[0:]
    data = np.stack(data, axis=0)
    return data

class trafic4cast_dataset(torch.utils.data.Dataset):
    def __init__(self, source_root, split_type='train',
                 cities=['BERLIN', 'ISTANBUL', 'MOSCOW'],
                 transform=None, reduce=False, num_frames=18, 
                 do_subsample=None, include_static = False):
        """Dataloader for the trafic4cast competition
        Usage Dataloader:
        The dataloader is situated in "videoloader.py", to use it, you have
        to download the competition data and set the "source_root" path.
        
        Args:
            source_root (str): Is the directory with the raw competition data.
            target_root (str, optional): This directory will be used to store the
                preprocessed data
            split_type (str, optional): Can be ['training', 'validation']
            cities (list, optional): This can be used to limit the data loader to a
                subset of cities. Has to be a list! Default is ['Berlin', 'Moscow', 'Istanbul']
            transform (None, optional): Transform applied to x before returning it.
            reduce (bool, optional): This option collapses the time dimension into the (color) channel dimension.
            num_frames (int, optional): number of frames in total
            include_static (bool, optional): the static data of each city will be appended to the end
            do_subsample (list of tuple, optional): List of two tuples. Returns only a part of the image. Slices the 
                image in the 'pixel' dimensions with x = x[n[0]:n[1], m[0]:m[1]]. with n,m as the tuples.
        """
        self.reduce = reduce
        self.root = source_root
        self.transform = transform
        self.split_type = split_type
        self.cities = cities
        self.include_static = include_static
        self.num_frames = num_frames
        self.subsample = False
        
        if do_subsample is not None:
            self.subsample = True
            self.n = do_subsample[0]
            self.m = do_subsample[1]

        file_paths = []
        
        for city in cities:
            file_paths = file_paths + glob.glob(os.path.join(self.root, city, self.split_type, '*.h5'))
        self.file_paths = file_paths

        
        if self.include_static:
            # read the static data of each city into memory 
            self.static = {}
            for city in cities:
                filepath = glob.glob(os.path.join(self.root, city, f'{city}_static_2019.h5'))[0]
                self.static[city] = load_h5_file(filepath)

    def __len__(self):
        return len(self.file_paths) * (288 - self.num_frames)
    
    def __getitem__(self, idx):
        """Summary
        Args:
            idx (TYPE): Description
        Returns:
            TYPE: Description
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file_ix = idx // (288 - self.num_frames)
        tstamp_ix = idx % (288 - self.num_frames)
        file_path = self.file_paths[file_ix]
        
        # access the file through indexing
        c_sample = h5py.File(file_path,'r')['array'][tstamp_ix: tstamp_ix + 18]
        x = np.moveaxis(c_sample[:12, :, :, :], -1, 1)
        y = np.moveaxis(c_sample[12:, :, :, :-1], -1, 1)

        if self.reduce:
            # stack all time dimensions into the channels.
            # all channels of the same timestamp are left togehter
            x = np.moveaxis(x, (0, 1), (2, 3))
            x = np.reshape(x, (495, 436, -1))

            if self.include_static:
                x = np.concatenate((x, self.static[city_name]), axis = -1)
            x = torch.from_numpy(x)
            x = x.permute(2, 0, 1)  # Dimensions: time/channels, h, w

            y = np.moveaxis(y, (0, 1), (2, 3))
            y = np.reshape(y, (495, 436, -1))
            y = torch.from_numpy(y)
            y = y.permute(2, 0, 1)
        else:
            if self.include_static:
                st = np.repeat(np.expand_dims(self.static[city_name], axis=0), 12, axis=0)
                st = np.moveaxis(st, -1, 1)
                x = np.concatenate((x, st), axis = 1)
            x = torch.from_numpy(x)
            y = torch.from_numpy(y)

        if self.subsample:
            x = subsample(x,self.n,self.m)
            y = subsample(y,self.n,self.m)
        
        if self.transform is not None:
            x =  self.transform(x)

        return x.float(), y.float()



def test_dataloader(train_loader):

    t_start = time.time()
    batch_size = train_loader.batch_size
    
    for batch_idx, (data, target) in enumerate(train_loader):
        print('batch_idx ', batch_idx)

        if batch_idx % 2 == 0:
            t_end = time.time()

            print('{} [{}/{} ({:.0f}%)]\t {:.0f}seconds \t{} - {}'.format(
                train_loader.dataset.split_type,
                batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), t_end-t_start,
                data.shape, target.shape))
            
            t_start = time.time()

        if batch_idx > 20:
            break

if __name__ == '__main__':

    source_root = r""

    kwds_dataset = {'reduce':True, 'return_features': False, 'return_city': False, 'include_static': False}    

    dataset_train = trafic4cast_dataset(source_root, split_type='training', cities=['BERLIN'],**kwds_dataset)
    dataset_val = trafic4cast_dataset(source_root, split_type='validation', cities=['BERLIN'], **kwds_dataset)

    kwds_train = {'shuffle': True, 'num_workers': 1,'batch_size': 4}
    kwds_val = {'shuffle': False, 'num_workers': 1,'batch_size': 4}
    train_loader = torch.utils.data.DataLoader(dataset_train, **kwds_train)
    val_loader = torch.utils.data.DataLoader(dataset_val, **kwds_val)

    kwds_tester = {}
    test_dataloader(train_loader, **kwds_tester)
    test_dataloader(val_loader, **kwds_tester)


