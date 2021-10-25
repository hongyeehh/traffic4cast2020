import numpy as np
import torch
import torch.nn as nn
import h5py
from pathlib import Path
import pickle
import sys, os, glob
import datetime as dt
from tqdm import tqdm

sys.path.append(os.getcwd())

from multiLSTM.config import config
from multiLSTM.convLSTM import Encoder, Forecaster, EF, CGRU_cell

from collections import OrderedDict

# please enter the source data root and submission root
source_root = r"D:/Traffic4/Data/2020/ori"
submission_root = r"D:/Traffic4/Data/2020/submit"

model_root = r"D:/Traffic4/runs/convLSTM_1634977731/checkpoint.pt"
mask_root = r"util/masks.dict"


convlstm_encoder_params = [
    [
        OrderedDict({"conv1_relu_1": [16, 16, 3, 1, 1], "pool_2": [2, 2, 0]}),
        OrderedDict({"conv2_relu_1": [16, 32, 3, 1, 1], "pool_2": [2, 2, 0]}),
        OrderedDict({"conv3_relu_1": [32, 96, 3, 1, 1], "pool_2": [2, 2, 0]}),
        OrderedDict({"conv4_relu_1": [96, 192, 3, 1, 1], "pool_2": [2, 2, 0]}),
    ],
    [
        CGRU_cell(input_channels=16, num_features=16, shape=(248, 224), filter_size=3),
        CGRU_cell(input_channels=32, num_features=32, shape=(124, 112), filter_size=3),
        CGRU_cell(input_channels=96, num_features=96, shape=(62, 56), filter_size=3),
        CGRU_cell(input_channels=192, num_features=192, shape=(31, 28), filter_size=3),
    ],
]


convlstm_forecaster_params = [
    [
        OrderedDict({"deconv1_relu_1": [192, 96, 3, 1, 1]}),
        OrderedDict({"deconv2_relu_1": [96, 32, 3, 1, 1]}),
        OrderedDict({"deconv3_relu_1": [32, 16, 3, 1, 1]}),
        OrderedDict(
            {"deconv4_relu_1": [16, 16, 3, 1, 1], "conv3_relu_2": [16, 16, 3, 1, 1], "conv3_3": [16, 8, 1, 1, 0]}
        ),
    ],
    [
        CGRU_cell(input_channels=192, num_features=192, shape=(31, 28), filter_size=3),
        CGRU_cell(input_channels=96, num_features=96, shape=(62, 56), filter_size=3),
        CGRU_cell(input_channels=32, num_features=32, shape=(124, 112), filter_size=3),
        CGRU_cell(input_channels=16, num_features=16, shape=(248, 224), filter_size=3),
    ],
]


def load_h5_file(file_path):
    """
    Given a file path to an h5 file assumed to house a tensor, 
    load that tensor into memory and return a pointer.
    """
    # load
    fr = h5py.File(file_path, "r")
    a_group_key = list(fr.keys())[0]
    data = list(fr[a_group_key])
    # transform to appropriate numpy array
    data = data[0:]
    data = np.stack(data, axis=0)
    return data


class WrappedModel(torch.nn.Module):
    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module  # that I actually define.

    def forward(self, x):
        return self.module(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1])
forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1])
model = EF(encoder, forecaster)
model = WrappedModel(model).to(device)
city = "Berlin"


mask_dict = pickle.load(open(mask_root, "rb"))

padd = torch.nn.ZeroPad2d((6, 6, 1, 0))

state_dict = torch.load(model_root, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# load static data
filepath = glob.glob(os.path.join(source_root, city, f"{city}_static_2019.h5"))[0]
static = load_h5_file(filepath)

static = torch.from_numpy(static).permute(2, 0, 1).unsqueeze(0).to(device).float()

# load mask
# mask_ = torch.from_numpy(mask_dict[city]["sum"] > 0).bool()

# get test data
file_paths = glob.glob(os.path.join(source_root, city, "testing", "*.h5"))
for path in tqdm(file_paths):
    # the date of the file
    all_data = load_h5_file(path)
    x = np.moveaxis(all_data, -1, 2)

    x = torch.from_numpy(x).to(device)
    x = torch.cat([x, static.repeat(x.shape[0], x.shape[1], 1, 1, 1)], axis=2)
    x = x / 255

    with torch.no_grad():
        inputs = padd(x)
        pred = model(inputs)

        # expand
        res = pred[:, :, :, 1:, 6:-6].cpu().float()

    # apply mask
    # masks = mask_.expand(res.shape)
    # res[~masks] = 0

    res = torch.clamp(res, 0, 255).permute(0, 1, 3, 4, 2).numpy().astype(np.uint8)

    # create saving root
    root = os.path.join(submission_root, city.upper())
    if not os.path.exists(root):
        os.makedirs(root)

    # save predictions
    target_file = os.path.join(root, path.split("\\")[-1])
    with h5py.File(target_file, "w", libver="latest",) as f:
        f.create_dataset("array", shape=(res.shape), data=res, compression="gzip", compression_opts=4)

