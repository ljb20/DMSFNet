from __future__ import absolute_import

from torchvision.transforms import *
import torch
#from PIL import Image
import random
import math
#import numpy as np
#import torch
class ChannelT(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, probability = 0.5):
        self.probability = probability

       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img
        else:
            a=random.uniform(0.01, 0.5)
            b=random.uniform(0.01, 0.5)
            c=random.uniform(0.01, 0.5)
            img[0, :,:] = 2*a*img[0,:,:]
            img[1, :,:] = 2*b*img[1,:,:]
            img[2, :,:] = 2*c*img[2,:,:]

        return img
   
class ChannelAdapGray(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, probability = 0.5):
        self.probability = probability

       
    def __call__(self, img):

        # if random.uniform(0, 1) > self.probability:
            # return img

        idx = random.randint(0, 1)
        
        if idx ==0:
        #     # random select R Channel
        #     img[1, :,:] = img[0,:,:]
        #     img[2, :,:] = img[0,:,:]
        # elif idx ==1:
        #     # random select B Channel
        #     img[0, :,:] = img[1,:,:]
        #     img[2, :,:] = img[1,:,:]
        # elif idx ==2:
        #     # random select G Channel
        #     img[0, :,:] = img[2,:,:]
        #     img[1, :,:] = img[2,:,:]
            img[0, :,:] = img[1,:,:]
            img[2, :,:] = img[1,:,:]
        else:
            if random.uniform(0, 1) > self.probability:
                # return img
                img = img
            else:
                tmp_img = 0.2989 * img[0,:,:] + 0.5870 * img[1,:,:] + 0.1140 * img[2,:,:]
                img[0,:,:] = tmp_img
                img[1,:,:] = tmp_img
                img[2,:,:] = tmp_img
        return img
  
class ChannelExchange(object):
    """ Adaptive selects a channel or two channels.
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    
    def __init__(self, gray = 2):
        self.gray = gray

    def __call__(self, img):
    
        idx = random.randint(0, self.gray)
        
        if idx ==0:
            # random select R Channel
            img[1, :,:] = img[0,:,:]
            img[2, :,:] = img[0,:,:]
        elif idx ==1:
            # random select B Channel
            img[0, :,:] = img[1,:,:]
            img[2, :,:] = img[1,:,:]
        elif idx ==2:
            # random select G Channel
            img[0, :,:] = img[2,:,:]
            img[1, :,:] = img[2,:,:]
        else:
            tmp_img = 0.2989 * img[0,:,:] + 0.5870 * img[1,:,:] + 0.1140 * img[2,:,:]
            img[0,:,:] = tmp_img
            img[1,:,:] = tmp_img
            img[2,:,:] = tmp_img
        return img