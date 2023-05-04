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

def save_branches(file,feature_list,max_batch=10,name=""):
    batch = 0
    DFs = []
    for array in file.iterate(library="numpy",filter_name=feature_list,step_size=1000):
        temp_array = pd.DataFrame()
        for feature in feature_list: 
            if feature == "trk_dz":
                continue    
            #for ievt in range(len(array)):     
            #    array[feature][ievt] = array[feature][ievt] #+ np.random.normal(loc=branches_dict[feature]['smear'][0],scale=branches_dict[feature]['smear'][1]*batch, size = len(array[feature][ievt]))
            temp_array[feature] = np.concatenate(array[feature]).ravel()
                     

        FH = predictFastHisto(array["trk_z0"],array["trk_pt"],array["trk_eta"],array["trk_phi"],phieta_regions[batch]) 
        trk_dz = []
        for ievt in range(len(FH)):
            #print( FH[ievt])
            trk_dz.append(array["trk_z0"][ievt] - FH[ievt])
        temp_array["trk_dz"] = np.concatenate(trk_dz).ravel()
        temp_array = temp_array.loc[~((temp_array['trk_eta'] >= phieta_regions[batch][1][0]) & (temp_array['trk_eta'] < phieta_regions[batch][1][1]) & (temp_array['trk_phi'] >= phieta_regions[batch][0][0]) & (temp_array['trk_phi'] <= phieta_regions[batch][0][1])),:]
        DFs.append(temp_array)
        #TTTrackDF = pd.concat([TTTrackDF,temp_array],ignore_index=False)
        print("Batch "+str(batch)+" Cumulative", name, "read: ", len(temp_array))
        del [temp_array]
        del [array]
        batch += 1
        if batch >= max_batch:
            break
    trackskept = feature_list

    dataframes = []

    for DF in DFs:
        dataframe = pd.DataFrame()
        Tracks = DF[trackskept]
        Tracks.reset_index(inplace=True)
        print(len(Tracks))
        Tracks.dropna(inplace=True)
        print(len(Tracks))
        del [DF]

        for j in trackskept:
            dataframe[j] = Tracks[j]
        del [Tracks]

        infs = np.where(np.asanyarray(np.isnan(dataframe)))[0]
        dataframe.drop(infs, inplace=True)
        print("Reading Complete, read: ", len(dataframe), name)
        dataframes.append(dataframe)

    gc.collect()

    return dataframes

def predictFastHisto(value,weight,eta,phi,mask):
    max_z0 = 150
    nbins = 256
    z0List = []
    halfBinWidth = 0.5*(2*max_z0)/nbins
    for ibatch in range(value.shape[0]):
        notdrop = (eta[ibatch] <= mask[1][0]) | (eta[ibatch] > mask[1][1]) | (phi[ibatch] <= mask[0][0]) | (phi[ibatch] > mask[0][1])
        hist,bin_edges = np.histogram(value[ibatch][notdrop],nbins,range=(-1*max_z0,max_z0),weights=weight[ibatch][notdrop])
        hist = np.convolve(hist,[1,1,1],mode='same')
        z0Index= np.argmax(hist)
        z0 = -1*max_z0 +(2*max_z0)*z0Index/nbins+halfBinWidth
        z0List.append([z0])
    return np.array(z0List,dtype=np.float32)

branches_dict = {'trk_pt' : {'name':'trk $p_T$ [GeV]',                   'range':(0,128),      'bins':50, 'smear':(0,1.467)},#(0,0.1467)
                 'trk_eta': {'name':'trk $\\eta$',                       'range':(-2.4,2.4),   'bins':50, 'smear':(0,3.209e-2)},#(0,3.209e-3)
                 'trk_phi': {'name':'trk $\\phi_0$ [rad]',               'range':(-3.14,3.14), 'bins':50, 'smear':(0,1.292e-2)},#(0,1.292e-3)
                 'trk_z0' : {'name':'trk $z_0$ [cm]',                    'range':(-150,150),   'bins':50, 'smear':(0,2.468)},#(0,2.468)
                 'trk_PU' : {'name':'trk PU',                            'range':(0,1),        'bins':50, 'smear':(0,0)},#(0,0)
                 'trk_dz' : {'name':'|$PV_{reco}$ - $PV_{truth}$| [cm]', 'range':(-150,150),   'bins':50, 'smear':(0,0)},#(0,0)
                }

phieta_regions = [((3.2,-3.2),(2.5,-2.5)),
                  ((0.0,0.698),(-2.5,2.5)),
                  ((-0.698,0.0),(-2.5,2.5)),
                  ((-0.698,0.698),(-2.5,2.5)),
                  ((0.0,0.1),(0.8,1.4)),
                  ((-0.698,0.0),(0.8,1.4)),
                  ((-3.2,3.2),(0.8,1.4)),
                  ((0.0,0.1),(-0.8,1.2)),
                  ((0.0,0.02),(-0.8,1.2)),
                  ((0.0,0.698),(-0.8,1.2))]

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
DFs = save_branches(events,branches_dict.keys(),max_batch=10,name="")
# Get random ordered indices of track DF, this will tell us 
# which tracks to pick out of total dataset when getting train, test, val
for i,DF in enumerate(DFs):
    perm = np.random.permutation(DF.index)

    # Find number of tracks in train fraction [0:train_end] is train fraction
    train_end = int(train_fraction * len(DF.index))
    # Find tracks for val fraction [train_end:validate_end] is val fraction
    validate_end = int(val_fraction * len(DF.index)) + train_end
    # Sample DF taking the training indices, randomised by permutation
    # same for validate and test
    train = DF.loc[perm[:train_end]]
    validate = DF.loc[perm[train_end:validate_end]]
    test = DF.loc[perm[validate_end:]]

    print("#=======================================#")
    print("|                                       |")
    print("|           Drop "+str(i)+"                     |")
    print("|                                       |")
    print("#=======================================#")

    print("================= Train =================")
    print(train.describe())
    print("================= Validate =================")
    print(validate.describe())
    print("================= Test =================")
    print(test.describe())

    # Reset indices, currently have original indices from DF
    train.reset_index(inplace=True)
    validate.reset_index(inplace=True)
    test.reset_index(inplace=True)

    # Save, can modify these locations as needed
    newname = name + "_drop_"+str(i)
    os.system("mkdir "+newname)
    os.system("mkdir "+newname+ "/Train")
    os.system("mkdir "+newname+ "/Test")
    os.system("mkdir "+newname+ "/Val")
    os.system("mkdir "+newname+ "/Plots")


    train.to_pickle(newname+"/Train/train.pkl") 
    validate.to_pickle(newname+"/Val/val.pkl") 
    test.to_pickle(newname+"/Test/test.pkl") 

    for branch in branches_dict.keys():
        plt.clf()
        figure = plot_split_histo(DF['trk_PU'], DF[branch],
                                branches_dict[branch]['name'],
                                branches_dict[branch]['range'],
                                branches_dict[branch]['bins'],)
        plt.savefig("%s/%s.png" % (newname+"/Plots", branch+"_histo"))
        plt.close()
