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
import gc
pd.options.mode.chained_assignment = None
# Random seed for dataset splitting
np.random.seed(42)

rootdir = "/home/cb719/Delphes/megasim/Delphes-3.5.0/"
f = sys.argv[1]
name = sys.argv[2]

# Which branches to create for training
# pv_reco is the standard FastHisto vertex run in CMSSW
# from_PV defines if a track is primary vertex, this is our target

def save_branches(file,dataframe,feature_list,max_batch=10000,name=""):
    batch = 0
    TTTrackDF = pd.DataFrame()
    for array in file.iterate(library="numpy",filter_name=feature_list,step_size=1000):
        temp_array = pd.DataFrame()
        for feature in feature_list: 
            if feature == "trk_dz":
                continue    
            for ievt in range(len(array)):     
                array[feature][ievt] = array[feature][ievt] + np.random.normal(loc=branches_dict[feature]['smear'][0],scale=branches_dict[feature]['smear'][1], size = len(array[feature][ievt]))
            temp_array[feature] = np.concatenate(array[feature]).ravel()
        FH = predictFastHisto(array["trk_z0"],array["trk_pt"]) 
        trk_dz = []
        for ievt in range(len(FH)):
            #print( FH[ievt])
            trk_dz.append(array["trk_z0"][ievt] - FH[ievt])
        temp_array["trk_dz"] = np.concatenate(trk_dz).ravel()
        TTTrackDF = pd.concat([TTTrackDF,temp_array],ignore_index=False)
        print("Cumulative", name, "read: ", len(TTTrackDF))
        del [temp_array]
        del [array]
        batch += 1
        if batch >= max_batch:
            break
    trackskept = feature_list

    Tracks = TTTrackDF[trackskept]
    Tracks.reset_index(inplace=True)
    Tracks.dropna(inplace=True)
    del [TTTrackDF]

    for j in trackskept:
        dataframe[j] = Tracks[j]
    del [Tracks]

    infs = np.where(np.asanyarray(np.isnan(dataframe)))[0]
    dataframe.drop(infs, inplace=True)
    print("Reading Complete, read: ", len(dataframe), name)

    gc.collect()

    return dataframe

def predictFastHisto(value,weight):
    max_z0 = 150
    nbins = 256
    z0List = []
    halfBinWidth = 0.5*(2*max_z0)/nbins
    for ibatch in range(value.shape[0]):
        hist,bin_edges = np.histogram(value[ibatch],nbins,range=(-1*max_z0,max_z0),weights=weight[ibatch])
        hist = np.convolve(hist,[1,1,1],mode='same')
        z0Index= np.argmax(hist)
        z0 = -1*max_z0 +(2*max_z0)*z0Index/nbins+halfBinWidth
        z0List.append([z0])
    return np.array(z0List,dtype=np.float32)

branches_dict = {'trk_pt' : {'name':'trk $p_T$ [GeV]',                   'range':(0,128),      'bins':50, 'smear':(0,1.467)},#(0,0.1467)
                 'trk_eta': {'name':'trk $\\eta$',                       'range':(-2.4,2.4),   'bins':50, 'smear':(0,3.209e-2)},#(0,3.209e-3)
                 'trk_phi': {'name':'trk $\\phi_0$ [rad]',               'range':(-3.14,3.14), 'bins':50, 'smear':(0,1.292e-2)},#(0,1.292e-3)
                 'trk_z0' : {'name':'trk $z_0$ [cm]',                    'range':(-150,150),   'bins':50, 'smear':(0,24.68)},#(0,2.468)
                 'trk_PU' : {'name':'trk PU',                            'range':(0,1),        'bins':50, 'smear':(0,0)},#(0,0)
                 'trk_dz' : {'name':'|$PV_{reco}$ - $PV_{truth}$| [cm]', 'range':(-150,150),   'bins':50, 'smear':(0,0)},#(0,0)
                }

# How big each chunk read should be, only impacts performance, all tracks are 
# recombined into a single pandas dataframe
chunkread = 5000
batch_num = 0
# How big is our training and validation fractions, rest is testing
# train_fraction + val_fraction + test_fraction = 1
train_fraction = 0.7
val_fraction = 0.1

events = uproot.open(rootdir+f+":eventTree;29")

print(events)
print(events.keys())
#Define blank dataframe for tracks
TrackDF = pd.DataFrame()
save_branches(events,TrackDF,branches_dict.keys(),max_batch=10000,name="")
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

for branch in branches_dict.keys():
    plt.clf()
    figure = plot_split_histo(TrackDF['trk_PU'], TrackDF[branch],
                              branches_dict[branch]['name'],
                              branches_dict[branch]['range'],
                              branches_dict[branch]['bins'],)
    plt.savefig("%s/%s.png" % (name+"/Plots", branch+"_histo"))
    plt.close()
