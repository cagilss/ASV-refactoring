import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
import lib.torch_ard as nn_ard

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class LCNNARD(nn.Module):

    def __init__(self, num_classes=2):
        super(LCNN, self).__init__()

        self.features = nn.Sequential(
            mfm(1, 8, 5, 1, 2, type=2),
            
            nn.BatchNorm2d(8), #
            #nn.Dropout2d(p=0.35), #

            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(8, 16, 3, 1, 1),
            
            nn.BatchNorm2d(16), #
            #nn.Dropout2d(p=0.35), #

            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(16, 32, 3, 1, 1),

            nn.BatchNorm2d(32), #
            #nn.Dropout2d(p=0.35), #

            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(32, 24, 3, 1, 1),
            group(24, 24, 3, 1, 1),

            nn.BatchNorm2d(24), #
            #nn.Dropout2d(p=0.35), #

            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )
        # self.block = nn.Sequential(
        #     mfm(960, 256, type=0), # MFCC -> 384, MFCC_delta -> [960], spect -> 12480  IMFCC -> 1872
        #     nn.Dropout()  
        # )

        # self.logits = nn.Linear(256, 2)

        self.block = nn.Sequential(
            nn.Conv1d(144, 60, 3, padding=1),
            nn.BatchNorm1d(60),
            nn.Conv1d(60, 30, 3, padding=1),
            nn.BatchNorm1d(30),
            nn.Conv1d(30, 2, 3, padding=1)
        )

        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.init_weight()

    def forward(self, x):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        x = self.features(x)
        x = x.view(x.size(0), x.size(1)*x.size(3), -1)
        
        x = self.block(x)
        logits = torch.mean(x, 2)

        logsoftmax = self.logsoftmax(logits)
        softmax = self.softmax(logits)

        uncertanity, prob, alpha, evidence = self.compute_uncertanity(
            relu_evidence=self.exp_evidence, logits=logits)

        return logits, logsoftmax, softmax, uncertanity, prob, alpha

    def compute_uncertanity(self, relu_evidence, logits):
        K        = 2
        evidence = relu_evidence(logits)
        alpha    = evidence + 1
        uncrtnty = K / torch.sum(alpha, dim=1) 
        prob     = alpha / torch.sum(alpha, dim=1).reshape(-1,1)
        return uncrtnty, prob, alpha, evidence
    
    def relu_evidence(self):
        return F.relu
    
    def exp_evidence_2(self):
        return torch.exp

    def exp_evidence(self, logits): 
        return torch.exp(torch.clamp(logits,-10,10))
    
    def softplus_evidence(self):
        return F.softplus

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight.data)
                # m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_uniform(m.weight.data)
                #m.bias.data.zero_()
 

class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn_ard.Conv2dARD(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        elif type == 2:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn_ard.LinearARD(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x