import torch
from torch import nn 
import torch.nn.functional as F

class LightResNet(nn.Module):
    def __init__(self):
        super(LightResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.block1 = ResNetBlock(8, 16,  True)
        self.mp = nn.MaxPool2d(2, stride=3, padding=1)
        self.mp_stride_1 = nn.MaxPool2d(2, stride=1, padding=1)
        self.block2 = ResNetBlock(16, 32,  False)
        self.block3 = ResNetBlock(32, 16, False)

        #self.block4 = ResNetBlock(16, 16, False)
        #self.block5 = ResNetBlock(16, 16, False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(16)
        self.dropout = nn.Dropout(0.7) 

        self.logsoftmax = nn.LogSoftmax(dim=1)

        #self.fc1 = nn.Linear(336, 64) # 128
        #self.fc2 = nn.Linear(64, 2)

        self.conv1_1 = nn.Conv1d(48, 64, 3, padding=1)
        self.conv1_2 = nn.Conv1d(64, 2, 3, padding=1)

        
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

        self.sigmoid = nn.Sigmoid()
        
        self.init_weight()
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(dim=1)
        out = self.conv1(x)

        out = self.block1(out)
        out = self.mp(out)
        out = self.block2(out)
        
        out = self.mp_stride_1(out) # 
        out = self.block3(out)
        
        #out = self.mp_stride_1(out) 
        
        #out = self.block4(out) 
        #out = self.bn(out) 
        #out = self.relu(out) 
        out = self.mp_stride_1(out) 
        #out = self.block4(out) 

        #out = out.view(batch_size, -1)
        out = self.dropout(out)


        #out = self.fc1(out)
        #out = self.relu(out)
        
        #logits = self.fc2(out) # binary or 2class format
        
        out = out.view(out.size(0), out.size(1)*out.size(3), -1)

        out = self.conv1_1(out)
        out = self.conv1_2(out)
        logits = torch.mean(out, 2)
            

        logsoftmax = self.logsoftmax(logits)
        softmax = self.softmax(logits)

        uncertanity, prob, alpha = self.compute_uncertanity(
            relu_evidence=self.exp_evidence, logits=logits)

        return logits, logsoftmax, softmax, uncertanity, prob, alpha
    
    def compute_uncertanity(self, relu_evidence, logits):
        K        = 2
        evidence = relu_evidence(logits)
        alpha    = evidence + 1
        uncrtnty = K / torch.sum(alpha, dim=1) 
        prob     = alpha / torch.sum(alpha, dim=1).reshape(-1,1)
        return uncrtnty, prob, alpha

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
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                m.bias.data.zero_()


class ResNetBlock(nn.Module):
    def __init__(self, in_depth, depth, first=False):
        super(ResNetBlock, self).__init__()
        #self.first = first
        self.conv1 = nn.Conv2d(in_depth, depth, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(depth)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(depth, depth, kernel_size=3, stride=3, padding=1)
        self.conv11 = nn.Conv2d(in_depth, depth, kernel_size=3, stride=3, padding=1)

        self.pre_bn = nn.BatchNorm2d(depth)

    def forward(self, x):
        # x is (B x d_in x T)
        prev = x
        prev_mp =  self.conv11(x)

        out = self.conv1(x)
        # out is (B x depth x T/2)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        # out is (B x depth x T/2)
        out = out + prev_mp

        #if not self.first:
        out = self.pre_bn(out)
        out = self.relu(out)

        return out
