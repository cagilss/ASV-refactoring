import torch
from config import cfg
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def KL(alpha):
    K = 2
    #beta = tf.constant(np.ones((1,K)))
    beta = torch.FloatTensor(np.ones((1, K))).to(device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)

    lnB = torch.lgamma(S_alpha) - \
            torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, 
                    keepdim=True) + lnB + lnB_uni.to(device)
    return kl


def mse_loss(p, alpha, global_step, annealing_step): 
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S

    A = torch.sum((p-m)**2, dim=1, keepdim=True)
    B = torch.sum(alpha*(S-alpha)/(S*S*(S+1)), dim=1, keepdim=True)

    annealing_coef = min(1.0, (global_step/annealing_step))
    
    alp = E*(1-p) + 1 
    C =  annealing_coef * self.KL(alp)
    return (A + B) + C


def loss_EDL(func=torch.digamma):
    def loss_func(p, alpha, global_step, annealing_step): 
        S = torch.sum(alpha, dim=1, keepdim=True)
        E = alpha - 1
        A = torch.sum(p * (func(S) - func(alpha)), dim=1, keepdim=True)
        annealing_coef = min(1.0, (global_step/annealing_step))
        alp = E*(1-p) + 1 
        B =  annealing_coef * self.KL(alp)
        return (A + B)
    return loss_func

