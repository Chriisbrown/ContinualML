import os
import pandas as pd
import torch
from torch.utils.data import Dataset


class TrackDataset(Dataset):
    def __init__(self, dataset_dir ):
        self.dataset_dir = dataset_dir
        self.dataframe = pd.read_pickle(dataset_dir)

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
        self.target_feature = ['from_PV']

        self.X_data = self.dataframe[self.training_features].to_numpy(dtype="float")
        self.y_data = self.dataframe[self.target_feature].to_numpy(dtype="float")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return self.X_data[idx] , self.y_data[idx]