import time
import os
import random
import torch
from collections import OrderedDict

config = dict()

##################################################################

# Please enter where raw data are stored.
config["source_dir"] = r"D:\Traffic4\Data\2020\ori"

config["debug"] = False
config["city"] = "Berlin"

config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##################################################################

# data loader configuration
config["num_workers"] = 18


# Hyper-parameters and training configuration.
config["batch_size"] = 6
config["learning_rate"] = 1e-2

# early stopping and lr schedule
config["patience"] = 3
config["lr_step_size"] = 1
config["lr_gamma"] = 0.1

config["num_epochs"] = 50

config["iters_to_accumulate"] = 1

if config["debug"] == False:
    config["print_every_step"] = 100
else:
    config["print_every_step"] = 5

