import uproot
import os
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

rootdir = "/home/cebrown/Documents/Datasets/VertexDatasets/OldKFGTTData_EmuTQ/"
f = sys.argv[1]
name = sys.argv[2]

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
                                 'real',
                                 'fake',
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
                           'real',
                           'fake',
                           ],
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
                          (0,1),
                          (0,1)
                          ],
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
                        2,
                        2
                        ]
}

# How big each chunk read should be, only impacts performance, all tracks are 
# recombined into a single pandas dataframe
chunkread = 5000
batch_num = 0
# How big is our training and validation fractions, rest is testing
# train_fraction + val_fraction + test_fraction = 1
train_fraction = 0.7
val_fraction = 0.1

events = uproot.open(rootdir+f+':L1TrackNtuple/eventTree')
#Define blank dataframe for tracks
TrackDF = pd.DataFrame()

#Iterate through events
for batch in events.iterate(step_size=chunkread, library='pd'):
    # Create some additional entries in batch dataframe, need to be same dimensions
    # as tracks not events. Jagged arrays prevent this being standardised
    # Iterate through events in batch
    batch[0]["real"] = (batch[0]["trk_fake"] != 0).astype(int)
    batch[0]["fake"] = (batch[0]["trk_fake"] == 0).astype(int)

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
os.system("mkdir "+name)
os.system("mkdir "+name+ "/Train")
os.system("mkdir "+name+ "/Test")
os.system("mkdir "+name+ "/Val")
os.system("mkdir "+name+ "/Plots")


train.to_pickle(name+"/Train/train.pkl") 
validate.to_pickle(name+"/Val/val.pkl") 
test.to_pickle(name+"/Test/test.pkl") 

skip_plotting = ["real","fake","trk_fake"]
for i in range(len(branches_dict['ntuple_names'])):
    plt.clf()
    if (branches_dict['ntuple_names'][i] in skip_plotting):
        pass
    figure = plot_split_histo(TrackDF['real'], TrackDF[branches_dict['ntuple_names'][i]],
                              branches_dict['names'][i],
                              branches_dict['ranges'][i],
                              branches_dict['bins'][i],)
    plt.savefig("%s/%s.png" % (name+"/Plots", branches_dict['ntuple_names'][i]+"_histo"))
    plt.close()
