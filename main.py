import os
import random 
from torch.optim import Adam 
from profilehooks import profile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
from utils.data import ASVDatasetTorch, asvdataset_collate_fn_pad, BinnedLengthSampler

# defaulth configuration settings
from config import cfg, update_config

# import models 
from model.lcnn import LCNN
from model.lcnn import LCNNARD 

from model.rnn import LSTMNeuralNetwork
from model.resnet import LightResNet

# import loss functions
from loss.functions import loss_EDL #edl or mse

from utils.trainer import train, validate

from dataclasses import dataclass
import pathlib

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')


@dataclass
class Data:
    tr_dataloader : DataLoader
    val_dataloader : DataLoader
    eval_dataloader : DataLoader


def run_session():    
    # generate neural network model 
    if cfg.TRAIN.MODEL == 'LSTM':
        model = LSTMNeuralNetwork().to(device)
    if cfg.TRAIN.MODEL == 'LCNN':
        model = LCNN().to(device)
    if cfg.TRAIN.MODEL == 'ResNet':
        model = LightResNet().to(device)

    # generate loss function
    if cfg.TRAIN.LOSS == 'CrossEntropy':
        loss = nn.CrossEntropyLoss().to(device)
    if cfg.TRAIN.LOSS == 'EDL':
        loss = loss_EDL(torch.digamma)
    if cfg.TRAIN.LOSS == 'ELBO':
        pass
    
    # generate optimizer 
    optimizer = Adam(model.parameters(), lr=cfg.TRAIN.LR, eps=0.1)

    # train from scratch 
    if cfg.MODE == 'train':
        print('model specs...')
        for epoch in range(cfg.TRAIN.EPOCH):        
            train(model=model, criterion=loss, optimizer=optimizer, 
                tr_dataloader=Data.tr_dataloader, epoch=epoch)

    # validate with the pre-trained model
    if cfg.MODE == 'validation':
        validate(model=model, loss_func=loss, optimizer=optimizer, 
            data=(Data.val_dataloader, Data.eval_dataloader), epoch=epoch)


def init_seed_and_cudnn(worker_id):
    torch.manual_seed(worker_id)
    torch.cuda.manual_seed_all(worker_id)
    np.random.seed(worker_id)
    random.seed(worker_id)

    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(worker_id)
    
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC


if __name__ == '__main__':
    init_seed_and_cudnn(cfg.SEED)

    ROOT = pathlib.Path(__file__).parent.resolve()

    # dir name of the cqcc features
    directory_name = 'cqcc_npy_norm'

    # root of the dataset
    directory_full_path = os.path.join(ROOT, directory_name)

    parser = argparse.ArgumentParser(description='Train energy network')
    
    # general configuration 
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=False,
                        type=str,
                        default=os.path.join('exps', 'verification', 'lcnn.yaml')
                        )
    
    parser.add_argument('--data_dir',
                        help='experiment configure file name',
                        required=False,
                        type=str,
                        default=directory_full_path
                        )

    args = parser.parse_args()

    # update default configuration with .yaml file 
    update_config(cfg, args)

    # dynamic padding settings
    batch_size = cfg.TRAIN.BATCH_SIZE
    bin_size = cfg.TRAIN.BIN_SIZE
    n_worker = cfg.TRAIN.N_WORKER

    tr_dataset = ASVDatasetTorch(
        os.path.join(directory_full_path, cfg.DATA.TRAIN_X),
        os.path.join(directory_full_path, cfg.DATA.TRAIN_Y)
        )

    val_dataset = ASVDatasetTorch(
        os.path.join(directory_full_path, cfg.DATA.DEV_X), 
        os.path.join(directory_full_path, cfg.DATA.DEV_Y)
        )

    eval_dataset = ASVDatasetTorch(
        os.path.join(directory_full_path, cfg.DATA.EVAL_X), 
        os.path.join(directory_full_path, cfg.DATA.EVAL_Y)
        )

    seq_lengths_tr = tr_dataset.get_seq_lengths()  
    seq_lengths_val = val_dataset.get_seq_lengths() 
    seq_lengths_eval = eval_dataset.get_seq_lengths()        

    sampler_tr = BinnedLengthSampler(seq_lengths_tr, batch_size, batch_size*bin_size)
    sampler_val = BinnedLengthSampler(seq_lengths_val, batch_size, batch_size*bin_size)
    sampler_eval = BinnedLengthSampler(seq_lengths_eval, batch_size, batch_size*bin_size)

    tr_dataloader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=False, 
                        sampler=sampler_tr, collate_fn=asvdataset_collate_fn_pad, 
                        num_workers=n_worker, pin_memory=True)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                         sampler=sampler_val, collate_fn=asvdataset_collate_fn_pad, 
                         num_workers=n_worker, pin_memory=True)

    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, 
                        sampler=sampler_eval, collate_fn=asvdataset_collate_fn_pad, 
                        num_workers=n_worker, pin_memory=True)

    #run_model([tr_dataloader, val_dataloader, eval_dataloader], args)

    # set Data class attribuıtes  
    Data.tr_dataloader = tr_dataloader
    Data.eval_dataloader = eval_dataloader
    Data.val_dataloader = val_dataloader
    
    run_session()
