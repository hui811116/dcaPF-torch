import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as tdist
import numpy as np
import sys

class Loss(nn.Module):
    def __init__(self, batch_size, class_num, device):
        super(Loss,self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum") # FIXME

    def kl_divergence(self,d1,d2,K):
        if (type(d1), type(d2)) in tdist.kl._KL_REGISTRY:
            return tdist.kl_divergence(d1,d2)
        else:
            samples = d1.rsample(torch.Size([K]))
            return (d1.log_prob(samples)-d2.log_prob(samples)).mean(0)
