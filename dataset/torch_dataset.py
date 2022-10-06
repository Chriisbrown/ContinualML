import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class TrackDataset(Dataset):
    """ TrackDataset pytorch dataset

    Takes dataset_dir as a string input

    """
    def __init__(self, dataset_dir ):
        self.dataset_dir = dataset_dir
        self.dataframe = pd.read_pickle(dataset_dir)

        # Training features used by model, for new ones, define them in dataset generator
        self.training_features = ['trk_MVA1', 
                                  'trk_bendchi2',
                                  'trk_chi2rphi', 
                                  'trk_chi2rz', 
                                  'trk_eta', 
                                  'trk_nstub', 
                                  'trk_phi',
                                  'trk_pt',
                                  'trk_z0',
                                  "pv_reco"]
        
        # Binary target features
        self.target_feature = ['from_PV']

        # Cast to numpy, quicker acces when getting items
        self.X_data = self.dataframe[self.training_features].to_numpy(dtype="float")
        self.y_data = self.dataframe[self.target_feature].to_numpy(dtype="float")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return self.X_data[idx] , self.y_data[idx]

#Define any transformations to the data here to simulate detector performance degredation on a track level
class Randomiser(object):
    """Generic Transformation of track data.

    Sets feature to random number

    Takes floating point max and string feature for training_features as input

    """

    def __init__(self,max,feature):
        self.max_int = max
        self.feature = feature

    def __call__(self, sample):
        sample[self.feature] = self.max*np.random.random()
        return sample