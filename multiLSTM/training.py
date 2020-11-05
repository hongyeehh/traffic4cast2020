import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
import numpy as np
import time
import pickle
import sys, os, random
from pathlib import Path
from tqdm import tqdm
random.seed(1)

sys.path.append(os.getcwd())

from videoloader import trafic4cast_dataset, test_dataloader
from config import config
from UNet import UNet
from visualizer import Visualizer
from earlystopping import EarlyStopping



city = config['city']
if config['debug'] == True:
    networkName = 'Test'
else:
    networkName = 'mLSTM_'


convlstm_encoder_params = [
    [
        OrderedDict({
            'conv1_relu_1': [8, 16, 5, 1, 2],
            'pool_2': [2, 2, 0]
            }),
        OrderedDict({
            'conv2_relu_1': [16, 32, 5, 1, 2],
            'pool_2': [2, 2, 0]
            }),
        OrderedDict({
            'conv3_relu_1': [32, 64, 3, 1, 1],
            'pool_2': [2, 2, 0]
            }),
        OrderedDict({
            'conv4_relu_1': [64, 96, 3, 1, 1],
            'pool_2': [2, 2, 0]
            }),
    ],

    [
        CGRU_cell(input_channels=16, num_features=16, shape=(248, 224), filter_size=5),
        CGRU_cell(input_channels=32, num_features=32, shape=(124, 112), filter_size=3),
        CGRU_cell(input_channels=64, num_features=64, shape=(62, 56), filter_size=3),
        CGRU_cell(input_channels=96, num_features=96, shape=(31, 28), filter_size=3),
    ]
]


convlstm_forecaster_params = [
    [
        OrderedDict({'deconv1_relu_1': [96, 64, 3, 1, 1]}),
        OrderedDict({'deconv2_relu_1': [64, 32, 3, 1, 1]}),
        OrderedDict({'deconv3_relu_1': [32, 16, 5, 1, 2]}),
        OrderedDict({
            'deconv4_relu_1': [16, 16, 5, 1, 2],
            'conv3_relu_2': [16, 16, 3, 1, 1],
            'conv3_3': [16, 8, 1, 1, 0]
        }),
    ],

    [
        CGRU_cell(input_channels=96, num_features=96, shape=(31, 28), filter_size=3),
        CGRU_cell(input_channels=64, num_features=64, shape=(62, 56), filter_size=3),
        CGRU_cell(input_channels=32, num_features=32, shape=(124, 112), filter_size=3),
        CGRU_cell(input_channels=16, num_features=16, shape=(248, 224), filter_size=5),
    ]
]

def trainNet(model, train_loader, val_loader, device, mask=None):
        
    print("=" * 30)

    # define the optimizer & learning rate 
    optim = torch.optim.SGD(model.parameters(), 
                            lr=config['learning_rate'], 
                            weight_decay=0.0001,
                            momentum=0.9,
                            nesterov=True)

    scheduler = StepLR(optim, step_size=config['lr_step_size'], gamma=config['lr_gamma'])

    log_dir = '../runs/' + networkName + str(int(datetime.now().timestamp()))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = Visualizer(log_dir)

    # Time for printing
    training_start_time = time.time()
    globaliter = 0
    scheduler_count = 0
    scaler = GradScaler()

    # initialize the early_stopping object
    early_stopping = EarlyStopping(log_dir, patience=config['patience'], verbose=True)

    # Loop for n_epochs
    for epoch in range(config['num_epochs']):
        writer.write_lr(optim, globaliter)

        # train for one epoch
        globaliter = train(model, train_loader, optim, device, writer, epoch, globaliter, scaler, mask)

        # At the end of the epoch, do a pass on the validation set
        val_loss = validate(model, val_loader, device, writer, epoch, mask)
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")            
            if scheduler_count == 2:
                break
            model.load_state_dict(torch.load(log_dir + '/checkpoint.pt'))
            early_stopping.early_stop = False
            early_stopping.counter = 0
            scheduler.step()
            scheduler_count += 1
        if config['debug'] == True:
            break
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

    model.load_state_dict(torch.load(log_dir + '/checkpoint.pt'))

    # remember to close the writer
    writer.close()

def train(model, train_loader, optim, device, writer, epoch, globaliter, scaler, mask=None):

    padd = torch.nn.ZeroPad2d((6,6,1,0))

    model.train()
    running_loss = 0.0
    n_batches = len(train_loader)

    # define start time
    start_time = time.time()
    optim.zero_grad()
    for i, (inputs, Y, _) in enumerate(train_loader, 0):
        # normalize
        inputs = inputs/255
        
        globaliter += 1
        inputs = padd(inputs[:,:,:8,...]).to(device)
        Y = Y.to(device)

        with autocast():
            out = model(inputs)

            if mask is not None:
                masks = padd(mask).expand(out.shape)
                out[~masks] = 0
            loss_size = torch.nn.functional.mse_loss(out[:,:,:,1:,6:-6], Y)/config['iters_to_accumulate']
        
        scaler.scale(loss_size).backward()
        
        if (i + 1) % config['iters_to_accumulate'] == 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
            
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()
        
        # Print statistics
        running_loss += loss_size.item() * config['iters_to_accumulate']
        if (i+1) % config['print_every_step'] == 0:
            print("Epoch {}, {:.1f}% \t train_loss: {:.3f} took: {:.2f}s".format(
                    epoch+1, 100 * (i+1) / n_batches, running_loss  / config['print_every_step'], time.time() - start_time))

            # write the train loss to tensorboard
            writer.write_loss_train(running_loss/ config['print_every_step'], globaliter)

            # Reset running loss and time
            running_loss = 0.0
            start_time = time.time()
        
        if config['debug'] == True and i == 200:
            break

    return globaliter


def validate(model, val_loader, device, writer, epoch, mask=None):

    padd = torch.nn.ZeroPad2d((6,6,1,0))

    random_visualize = random.randint(0, len(val_loader))
    if config['debug'] == True:
        random_visualize = 0

    total_val_loss = 0
    # change to validation mode
    model.eval()
    with torch.no_grad():
        for i, (val_inputs, val_y, _) in tqdm(enumerate(val_loader, 0)):
            val_inputs = val_inputs/255

            val_inputs = padd(val_inputs[:,:,:8,...]).to(device)
            val_y = val_y.to(device)
            
            out = model(val_inputs)
            if mask is not None:
                masks = padd(mask).expand(out.shape)
                out[~masks] = 0
            val_loss_size = torch.nn.functional.mse_loss(out[:,:,:,1:,6:-6], val_y)
            
            total_val_loss += val_loss_size.item()

            # each epoch select one prediction set (one batch) to visualize
            if i == random_visualize:
                writer.write_video(out.cpu(), epoch, if_predict=True)
                writer.write_video(val_y.cpu(), epoch, if_predict=False)

    val_loss = total_val_loss / len(val_loader)
    print("Validation loss = {:.2f}".format(val_loss))
    # write the validation loss to tensorboard
    writer.write_loss_validation(val_loss , epoch)
    return val_loss

if __name__ == "__main__":
    
    dataset_train = trafic4cast_dataset(source_root=config['source_dir'],
                                        split_type='training',
                                        cities=[city], 
                                        reduce=False
                                        )
    dataset_val = trafic4cast_dataset(source_root=config['source_dir'],
                                        split_type='validation',
                                        cities=[city], 
                                        reduce=False
                                        )
    
    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=config['batch_size'],
                                               shuffle=True,
                                               num_workers=config['num_workers'])
    val_loader = torch.utils.data.DataLoader(dataset_val,
                                             batch_size=config['batch_size'], 
                                             shuffle=False,
                                             num_workers=config['num_workers'])
    
    device = config['device']
    
    encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1])
    forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1])
    model = EF(encoder, forecaster)

    model = nn.DataParallel(model)
    model.to(device)

    # please enter the mask dir
    mask_dir = ''
    mask_dict = pickle.load(open( mask_dir, "rb" ) )
    mask_ = torch.from_numpy(mask_dict[city]['sum'] > 0).bool()
    
    # get the trainable paramters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("# of parameters: ", params)

    trainNet(model, train_loader, val_loader, device, mask_)

