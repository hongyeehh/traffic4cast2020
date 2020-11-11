import time
import os
import random

config = dict()

##################################################################

# Where raw data are stored.
config['source_dir'] = '/data/2020traffic4/ori'
# config['source_dir'] = r'D:\Projects\traffic4\data\ori'

config['debug'] = False
##################################################################

# model statistics 
config['in_channels'] = 115
config['n_classes'] = 48
config['depth'] = 5
config['width'] = 7
config['if_padding'] = True
# up_mode (str): 'upconv' or 'upsample'.
config['up_mode'] = 'upconv'

# data loader configuration
config['num_workers'] = 1
# Hyper-parameters and training configuration.
config['batch_size'] = 4
config['learning_rate'] = 1e-2

# early stopping and lr schedule
config['patience'] = 3
config['lr_step_size'] = 1
config['lr_gamma'] = 0.1

config['num_epochs'] = 50
if config['debug'] == True:
	config['print_every_step'] = 1
else:
	config['print_every_step'] = 50
config['iters_to_accumulate'] = 2