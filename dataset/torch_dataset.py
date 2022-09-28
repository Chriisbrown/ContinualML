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

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        #print(self.dataframe[self.training_features].loc[idx].to_numpy(),self.dataframe[self.target_feature].loc[idx].to_numpy())
        return self.dataframe[self.training_features].loc[idx].to_numpy(dtype="float"), self.dataframe[self.target_feature].loc[idx].to_numpy(dtype="float")