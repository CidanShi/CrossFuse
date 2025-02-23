import random
import numpy as np
import torch.nn as nn
from torchvision import transforms as T
import torch


class Augment(nn.Module):
    def __init__(self):
        super(AugOpt, self).__init__()
        self.aggr_aug = T.GaussianBlur(kernel_size=(9, 21), sigma=(0.1, 5))
        self.weak_aug = T. RandomCrop(32)

    def forward(self, img):
        weak_aug_img = self.weak_aug(img)
        aggr_aug_img = self.aggr_aug(weak_aug_img)

        return weak_aug_img, aggr_aug_img
