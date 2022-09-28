import uproot
import numpy as np
import math
from math import isnan
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)

f = "/home/cebrown/Documents/Datasets/VertexDatasets/OldKF_test.root"

branches = [
    'trk_MVA1', 
    'trk_bendchi2',
    'trk_chi2rphi', 
    'trk_chi2rz', 
    'trk_eta', 
    'trk_fake', 
    'trk_nstub', 
    'trk_phi',
    'trk_pt',
    'trk_z0',
    "pv_reco",
    'pv_truth',
    'from_PV'
]


chunkread = 5000
batch_num = 0

events = uproot.open(f+':L1TrackNtuple/eventTree')

TrackDF = pd.DataFrame()

for batch in events.iterate(step_size=chunkread, library='pd'):
    batch[0]['pv_reco'] = batch[0]['trk_MVA1']
    batch[0]['pv_truth'] = batch[0]['trk_MVA1']
    for ievt in range(len(batch[0].reset_index(level=1).index.value_counts())):
        pvs = np.ones(len(batch[0]["trk_pt"][batch_num*chunkread + ievt]))
        pvs.fill(batch[2]['pv_L1reco'][batch_num*chunkread + ievt][0])
        batch[0]["pv_reco"][batch_num*chunkread + ievt] = pvs

        pv_t = np.ones(len(batch[0]["trk_pt"][batch_num*chunkread + ievt]))
        pv_t.fill(batch[2]['pv_MC'][batch_num*chunkread + ievt][0])
        batch[0]["pv_truth"][batch_num*chunkread + ievt] = pv_t
        

        batch[0]["from_PV"] = (batch[0]["trk_fake"] == 1).astype(int)

    batch_num += 1

    TrackDF = pd.concat([TrackDF,batch[0][branches]])

    if batch_num > 1:
        break

TrackDF.reset_index(inplace=True)
TrackDF.dropna(inplace=True)

train_fraction = 0.7
val_fraction = 0.1
print(TrackDF.head())

perm = np.random.permutation(TrackDF.index)

train_end = int(train_fraction * len(TrackDF.index))
validate_end = int(val_fraction * len(TrackDF.index)) + train_end
train = TrackDF.loc[perm[:train_end]]
validate = TrackDF.loc[perm[train_end:validate_end]]
test = TrackDF.loc[perm[validate_end:]]


print("================= Train =================")
print(train.describe())
print("================= Validate =================")
print(validate.describe())
print("================= Test =================")
print(test.describe())

train.reset_index(inplace=True)
validate.reset_index(inplace=True)
test.reset_index(inplace=True)

train.to_pickle("Train/train.pkl") 
validate.to_pickle("Val/val.pkl") 
test.to_pickle("Test/test.pkl") 
