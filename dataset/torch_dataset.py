import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class TrackDataset(Dataset):
    """ TrackDataset pytorch dataset

    Takes dataset_dir as a string input

    """
    # Training features used by model, for new ones, define them in dataset generator
    training_features = [#'trk_MVA1', 
                         'trk_bendchi2',
                         'trk_chi2rphi', 
                         'trk_chi2rz', 
                         'trk_eta', 
                         'trk_nstub', 
                         'trk_phi',
                         'trk_pt',
                         'trk_z0',
                         ]
        
        # Binary target features
    target_feature = ['real']

    def __init__(self, dataset_dir, transform=None ):
        self.dataset_dir = dataset_dir
        self.dataframe = pd.read_pickle(dataset_dir)
        self.transform = transform

        if self.transform:
            self.dataframe = self.transform(self.dataframe)

        # Cast to numpy, quicker acces when getting items
        self.X_data = self.dataframe[self.training_features].to_numpy(dtype='float')
        self.targets = self.dataframe[self.target_feature].to_numpy(dtype='float')
        

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return self.X_data[idx] , self.targets[idx]

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