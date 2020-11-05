import numpy as np
import torch
import h5py
from pathlib import Path
import pickle
import sys, os, glob
import datetime as dt
from tqdm import tqdm

sys.path.append(os.getcwd())

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



cities = ['Berlin','Moscow', 'Istanbul']

# please enter the source data root and submission root
source_root = r""
submission_root = r""

for city in cities:
    
    # get the test files
    file_paths = glob.glob(os.path.join(source_root, city, 'test', '*.h5'))
    for path in tqdm(file_paths):
        all_data = load_h5_file(path)
        all_data = torch.from_numpy(np.moveaxis(all_data,-1,2)).float()
        
        pred = torch.mean(all_data, dim=1).unsqueeze(1).repeat(1,6,1,1,1)
        pred = torch.clamp(pred, 0, 255).permute(0, 1, 3, 4, 2).astype(np.uint8)
        
        # create saving root
        root = os.path.join(submission_root, city.upper())
        if not os.path.exists(root):
            os.makedirs(root)
            
        # save predictions
        target_file = os.path.join(root, path.split('/')[-1])
        with h5py.File(target_file, 'w', libver='latest',) as f:
            f.create_dataset('array', shape = (pred.shape), data=pred, compression="gzip", compression_opts=4)