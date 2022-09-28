import uproot
import numpy as np
import math
from math import isnan
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


f = "OldKF_test.root"

branches = [
    'trk_MVA1', 
    'trk_bendchi2',
    'trk_chi2rphi', 
    'trk_chi2rz', 
    'trk_eta', 
    'trk_fake', 
    'trk_genuine', 
    'trk_nstub', 
    'trk_phi',
    'trk_pt',
    'trk_z0',
    "pv_reco",
    'pv_truth'
]


chunkread = 5000
batch_num = 0

events = uproot.open(f+':L1TrackNtuple/eventTree')

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

    print(batch[0][branches])
    batch_num += 1

    
