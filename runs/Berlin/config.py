import time
import os
import random
import torch

config = dict()

##################################################################

# Where raw data are stored.
config['source_dir'] = '/data/2020traffic4/ori'
# config['source_dir'] = r'D:\Work\Projects\traffic4\data\ori'

config['debug'] = False
config['city'] = 'Moscow'

config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
##################################################################

# model statistics 
config['in_channels'] = 115
config['n_classes'] = 48

# data loader configuration
config['num_workers'] = 1
# Hyper-parameters and training configuration.
config['batch_size'] = 4
config['learning_rate'] = 1e-2

# early stopping and lr schedule
config['patience'] = 3
config['lr_step_size'] = 1
config['lr_gamma'] = 0.1

config['num_epochs'] = 2
if config['debug'] == True:
	config['print_every_step'] = 5
else:
	config['print_every_step'] = 100
config['iters_to_accumulate'] = 1