from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch
import numpy as np


class Visualizer():

    def __init__(self, log_dir):

        self.summary_writer = SummaryWriter(log_dir=log_dir)
    
    def write_lr(self, optim, globaliter):
        for i, param_group in enumerate(optim.param_groups):
            self.summary_writer.add_scalar('learning_rate/lr_'+ str(i) , param_group['lr'], globaliter)
        self.summary_writer.flush()
    
    def write_loss_train(self, value, globaliter):
        self.summary_writer.add_scalar('Loss/train', value, globaliter)
        self.summary_writer.flush()

    def write_loss_validation(self, value, globaliter, if_testtimes=False):
        if if_testtimes:
            postfix = '_testtimes'
        else:
            postfix = ''

        self.summary_writer.add_scalar('Loss/validation'+postfix, value, globaliter)
        self.summary_writer.flush()
    
    def write_image(self, images, epoch, if_predict=False, if_testtimes=False):

        if if_testtimes:
            postfix = '_testtimes'
        else:
            postfix = ''
        if len(images.shape) == 4:
            batch_1 = images[0,:9,:,:].unsqueeze(1)
            batch_2 = images[0,9:18,:,:].unsqueeze(1)
            batch_3 = images[0,18:,:,:].unsqueeze(1)

        if if_predict:
            batch_1 = torchvision.utils.make_grid(batch_1, normalize=True, range=(0, 1))
            self.summary_writer.add_image('prediction'+postfix+"/step1", batch_1, epoch)
            batch_2 = torchvision.utils.make_grid(batch_2, normalize=True, range=(0, 1))
            self.summary_writer.add_image('prediction'+postfix+"/step2", batch_2, epoch)
            batch_3 = torchvision.utils.make_grid(batch_3, normalize=True, range=(0, 1))
            self.summary_writer.add_image('prediction'+postfix+"/step3", batch_3, epoch)
        else:
            batch_1 = torchvision.utils.make_grid(batch_1, normalize=True, range=(0, 1))
            self.summary_writer.add_image('groundTruth'+postfix+"/step1", batch_1, epoch)
            batch_2 = torchvision.utils.make_grid(batch_2, normalize=True, range=(0, 1))
            self.summary_writer.add_image('groundTruth'+postfix+"/step2", batch_2, epoch)
            batch_3 = torchvision.utils.make_grid(batch_3, normalize=True, range=(0, 1))
            self.summary_writer.add_image('groundTruth'+postfix+"/step3", batch_3, epoch)

        self.summary_writer.flush()
    
    def write_video(self, videos, epoch, if_predict=False, if_testtimes=False):
        batch_1 = videos[0,:,:,:].unsqueeze(0)
        if if_testtimes:
            postfix = '_testtimes'
        else:
            postfix = ''
        videos = torch.reshape(batch_1, (1, 6, 8, batch_1.shape[-2], batch_1.shape[-1]))
        if if_predict:
            self.summary_writer.add_video('prediction'+postfix, videos, epoch)
        else:
            self.summary_writer.add_video('groundTruth'+postfix, videos, epoch)
        self.summary_writer.flush()

    def close(self):
        self.summary_writer.close()