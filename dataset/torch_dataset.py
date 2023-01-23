import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class EventDataset(Dataset):
    """ TrackDataset pytorch dataset

    Takes dataset_dir as a string input

    """

    def __init__(self, dataset_dir, transform=None ):
        self.dataset_dir = dataset_dir
        self.X = np.load(dataset_dir+"X.npy")
        self.y = np.load(dataset_dir+"y.npy")
        self.transform = transform

        if self.transform:
            self.X = self.transform(self.X)
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return np.expand_dims(self.X[idx],axis=0) , np.expand_dims(self.y[idx],axis=0)

#Define any transformations to the data here to simulate detector performance degredation on a track level
class Randomiser(object):
    """Generic Transformation of track data.

    Sets feature to random number

    Takes floating point max and string feature for training_features as input

    """

    def __init__(self,max,feature):
        self.max = max
        self.feature = feature

    def __call__(self, sample):
        sample[self.feature] = self.max*np.random.random()
        return sample


class GaussianSmear(object):
    """Generic Transformation of track data.

    Sets feature to random number

    Takes floating point max and string feature for training_features as input

    """

    def __init__(self,mean,std,feature):
        self.mean = mean
        self.std = std
        self.feature = feature

    def __call__(self, sample):
        sample[self.feature] = np.random.normal(loc=self.mean,scale=self.std)+sample[self.feature] 
        return sample