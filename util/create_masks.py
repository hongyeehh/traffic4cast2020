import torch
import numpy as np
import time
import pickle

import sys, os
sys.path.append(os.getcwd())
from videoloader import trafic4cast_dataset, test_dataloader

# please enter the source data root
source_root = r'' 

mask_dict = {}
mask_dict_target = {}

# the final mask includes sum(pixels)>threshold 
threshold = 0

cities = ['Berlin','Moscow', 'Istanbul']
for city in cities:
    
    overall_sum = 0
    overall_sum_target = 0
    overall_sum_channel = 0
    logging = []
    
    kwds_dataset = {'cities': [city]}
    kwds_loader = {'shuffle': False, 'num_workers':4, 'batch_size':8}

    dataset_train = trafic4cast_dataset(source_root, split_type='training', **kwds_dataset)
    dataset_val = trafic4cast_dataset(source_root, split_type='validation', **kwds_dataset)
    
    train_loader = torch.utils.data.DataLoader(dataset_train, **kwds_loader)
    val_loader = torch.utils.data.DataLoader(dataset_val, **kwds_loader)
    
    loader_list = [train_loader, val_loader]
    # sum up all available data for a city
    for loader in loader_list:

        for batch_idx, (data, Y, context) in enumerate(loader):
            
            batch_sum = torch.sum(data, (0,1,2))
            overall_sum = overall_sum + batch_sum 

            if (batch_idx+1) % 1 == 0:

                batch_mask = (batch_sum  > threshold)
                overall_mask = (overall_sum  > threshold)
                
                nonzeros = np.sum(overall_mask.numpy())/overall_mask.numel() * 100
                print('{}, {} [{}/{}] {:.2f}% non-zeros'.format(
                        city, loader.dataset.split_type,
                        batch_idx * len(data), len(loader.dataset),
                        nonzeros))
            
    overall_sum = overall_sum.numpy()
    overall_mask = (overall_sum > threshold)

    mask_dict[city] = {'mask': overall_mask, 'sum': overall_sum}

    print(np.sum(overall_mask)/overall_mask.size,"% non-zeros in ", city)
    
print('writing file')
pickle.dump( mask_dict, open( os.path.join('.',"util","masks.dict"), "wb" ) )
print('done')
