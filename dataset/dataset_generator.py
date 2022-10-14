import uproot
import numpy as np
import math
from math import isnan
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from eval import *
pd.options.mode.chained_assignment = None
# Random seed for dataset splitting
np.random.seed(42)

f = sys.argv[1]

# Which branches to create for training
# pv_reco is the standard FastHisto vertex run in CMSSW
# from_PV defines if a track is primary vertex, this is our target

branches_dict = {'ntuple_names':['trk_MVA1', 
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
                                 'from_PV',
                                 'not_from_PV',
                                 'delta_z0'
                                # my_extra_features
                            ],
                  "names":['trk MVA1', 
                           'trk $\\chi^2_{bend}$',
                           'trk $\\chi^2_{r\\phi}$', 
                           'trk $\\chi^2_{rz}$', 
                           'trk $\\eta$', 
                           'trk fake', 
                           'trk #stub', 
                           'trk $\\phi$ [rad]',
                           'trk $p_T$ [GeV]',
                           'trk $z_0$ [cm]',
                           "$PV_{reco}$ [cm]",
                           '$PV_{truth}$ [cm]',
                           'from PV',
                           'not_from_PV',
                           '|$PV_{reco}$ - $PV_{truth}$| [cm]'],
                "ranges":[(0,1), 
                          (0,6),
                          (0,20), 
                          (0,20), 
                          (-2.4,2.4), 
                          (0,2), 
                          (4,7), 
                          (-3.14,3.14),
                          (0,127),
                          (-15,15),
                          (-15,15),
                          (-15,15),
                          (0,1),
                          (0,1),
                          (0,1)],
                "bins":[50, 
                        50,
                        50, 
                        50, 
                        50, 
                        3, 
                        3, 
                        20,
                        127,
                        50,
                        50,
                        50,
                        2,
                        2,
                        50]
}

# How big each chunk read should be, only impacts performance, all tracks are 
# recombined into a single pandas dataframe
chunkread = 5000
batch_num = 0
# How big is our training and validation fractions, rest is testing
# train_fraction + val_fraction + test_fraction = 1
train_fraction = 0.7
val_fraction = 0.1

events = uproot.open(f+':L1TrackNtuple/eventTree')
#Define blank dataframe for tracks
TrackDF = pd.DataFrame()

#Iterate through events
for batch in events.iterate(step_size=chunkread, library='pd'):
    # Create some additional entries in batch dataframe, need to be same dimensions
    # as tracks not events. Jagged arrays prevent this being standardised
    batch[0]['pv_reco'] = batch[0]['trk_MVA1']
    batch[0]['pv_truth'] = batch[0]['trk_MVA1']
    # Iterate through events in batch
    for ievt in range(len(batch[0].reset_index(level=1).index.value_counts())):
        #Create blank array for PV position same length as track data for this event
        pvs = np.ones(len(batch[0]["trk_pt"][batch_num*chunkread + ievt]))
        # Fill array with pv_l1Reco for event track in event
        pvs.fill(batch[2]['pv_L1reco'][batch_num*chunkread + ievt][0])
        # Put array into batch dataframe
        batch[0]["pv_reco"][batch_num*chunkread + ievt] = pvs

        # Repeat for truth level vertex position
        pv_t = np.ones(len(batch[0]["trk_pt"][batch_num*chunkread + ievt]))
        pv_t.fill(batch[2]['pv_MC'][batch_num*chunkread + ievt][0])
        batch[0]["pv_truth"][batch_num*chunkread + ievt] = pv_t

        # trk_fake defines tracks as 0 for fake, 1 for PV and 2 for PU
        # Need single varible for training so cast trk_fake==1 as int
        batch[0]["from_PV"] = (batch[0]["trk_fake"] == 1).astype(int)
        batch[0]["not_from_PV"] = (batch[0]["trk_fake"] != 1).astype(int)

        batch[0]["delta_z0"] = abs(batch[0]["pv_reco"] - batch[0]["pv_truth"])

        batch[0]['trk_eta'] = batch[0]['trk_eta']+ np.random.normal(loc=0,scale=5, size = len(batch[0]['trk_eta']) )
        batch[0]['trk_pt'] = batch[0]['trk_pt'] +  np.random.normal(loc=0,scale=0.1, size = len(batch[0]['trk_pt']))
        batch[0]['trk_z0'] = batch[0]['trk_z0'] +  np.random.normal(loc=5,scale=1, size = len(batch[0]['trk_z0']) )
        batch[0]['pv_reco'] = batch[0]['pv_reco'] +  np.random.normal(loc=5,scale=1, size = len(batch[0]['pv_reco']) )
        ##############################################################

        # Define other training features here and add name to branches list

        # batch[0]["my_feautre"] = batch[0]["pv_reco"] ** 2

        ##############################################################

    batch_num += 1

    #Add batch dataframe to larger dataframe, only defined branches are added to save memory
    TrackDF = pd.concat([TrackDF,batch[0][branches_dict['ntuple_names']]])
    print(batch_num," out of: ", len(events))

# Reset index due to double indices in concatanated array
TrackDF.reset_index(inplace=True)
# Remove any tracks with NA entries
TrackDF.dropna(inplace=True)

# Get random ordered indices of track DF, this will tell us 
# which tracks to pick out of total dataset when getting train, test, val
perm = np.random.permutation(TrackDF.index)

# Find number of tracks in train fraction [0:train_end] is train fraction
train_end = int(train_fraction * len(TrackDF.index))
# Find tracks for val fraction [train_end:validate_end] is val fraction
validate_end = int(val_fraction * len(TrackDF.index)) + train_end
# Sample TrackDF taking the training indices, randomised by permutation
# same for validate and test
train = TrackDF.loc[perm[:train_end]]
validate = TrackDF.loc[perm[train_end:validate_end]]
test = TrackDF.loc[perm[validate_end:]]

print("================= Train =================")
print(train.describe())
print("================= Validate =================")
print(validate.describe())
print("================= Test =================")
print(test.describe())

# Reset indices, currently have original indices from TrackDF
train.reset_index(inplace=True)
validate.reset_index(inplace=True)
test.reset_index(inplace=True)

# Save, can modify these locations as needed
train.to_pickle("Train/train.pkl") 
validate.to_pickle("Val/val.pkl") 
test.to_pickle("Test/test.pkl") 

skip_plotting = ["from_PV","not_from_PV","pv_truth","pv_reco","trk_fake"]
for i in range(len(branches_dict['ntuple_names'])):
    plt.clf()
    if (branches_dict['ntuple_names'][i] in skip_plotting):
        pass
    figure = plot_split_histo(TrackDF['from_PV'], TrackDF[branches_dict['ntuple_names'][i]],
                              branches_dict['names'][i],
                              branches_dict['ranges'][i],
                              branches_dict['bins'][i],)
    plt.savefig("%s/%s.png" % ("../eval/plots", branches_dict['ntuple_names'][i]+"_histo"))
    plt.close()
