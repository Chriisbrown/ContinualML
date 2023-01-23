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
from sklearn.model_selection import train_test_split

# Random seed for dataset splitting
np.random.seed(42)

def CreateHisto(value,weight, res_func, return_index = False, nbins=256, max_z0=15,factor=1):

    hist,bin_edges = np.histogram(value,nbins,range=(-1*max_z0,max_z0),weights=weight*res_func,density=True)
    hist = np.clip(hist,0,1)
    
    return hist/factor,bin_edges

def linear_res_function(x,return_bool = False):
        if return_bool:
            return np.full_like(x,True).astype(bool)
        else:
            return np.ones_like(x)

rootdir = "/home/cebrown/Documents/Datasets/VertexDatasets/OldKFGTTData_EmuTQ/"
f = sys.argv[1]
name = sys.argv[2]

# Which branches to create for training
# pv_reco is the standard FastHisto vertex run in CMSSW
# from_PV defines if a track is primary vertex, this is our target

nfeatures = 10
nbins = 256
max_z0 = 20.46912512

histo_names = ["$p_T$ ","$\eta$ ","MVA ","$\chi^2_{R\phi}$ ","$\chi^2_{rz}$ ","$\chi^2_{bend}$ ","$\phi$ ","# stub ","$\\frac{1}{\eta^2}$ ","Tracks","Vertex"][::-1]


# How big each chunk read should be, only impacts performance, all tracks are 
# recombined into a single pandas dataframe
chunkread = 5000
batch_num = 0
num_histos = 0
 
# How big is our training and validation fractions, rest is testing
# train_fraction + val_fraction + test_fraction = 1
train_fraction = 0.7
val_fraction = 0.1

events = uproot.open(rootdir+f+':L1TrackNtuple/eventTree')
num_events = events.num_entries
#Define blank dataframe for tracks
Histograms = np.ndarray([num_events,nfeatures,nbins])
Vertices = np.ndarray([num_events])
#Iterate through events
for batch in events.iterate(step_size=chunkread, library='pd'):

    # Create some additional entries in batch dataframe, need to be same dimensions
    # as tracks not events. Jagged arrays prevent this being standardised    # Iterate through events in batch
    for ievt in range(len(batch[0].reset_index(level=1).index.value_counts())):

        print("Event: ", ievt, " out of ", len(batch[0].reset_index(level=1).index.value_counts()))

        trk_overEta = 1/(0.1+0.2*(batch[0]['trk_eta'][batch_num*chunkread + ievt])**2)
        ntrk = (batch[0]['trk_fake'][batch_num*chunkread + ievt] >= 0).astype(int)

        pt_histo = CreateHisto(batch[0]['trk_z0'][batch_num*chunkread + ievt],batch[0]['trk_pt'][batch_num*chunkread + ievt],res_func=linear_res_function(batch[0]['trk_pt'][batch_num*chunkread + ievt]), nbins=nbins, max_z0=max_z0)
        eta_histo = CreateHisto(batch[0]['trk_z0'][batch_num*chunkread + ievt],abs(batch[0]['trk_eta'][batch_num*chunkread + ievt]),res_func=linear_res_function(batch[0]['trk_pt'][batch_num*chunkread + ievt]), nbins=nbins, max_z0=max_z0)
        MVA_histo = CreateHisto(batch[0]['trk_z0'][batch_num*chunkread + ievt],batch[0]['trk_MVA1'][batch_num*chunkread + ievt],res_func=linear_res_function(batch[0]['trk_pt'][batch_num*chunkread + ievt]), nbins=nbins, max_z0=max_z0)

        chi2rphi_histo = CreateHisto(batch[0]['trk_z0'][batch_num*chunkread + ievt],batch[0]['trk_chi2rphi'][batch_num*chunkread + ievt],res_func=linear_res_function(batch[0]['trk_pt'][batch_num*chunkread + ievt]), nbins=nbins, max_z0=max_z0)
        chi2rz_histo = CreateHisto(batch[0]['trk_z0'][batch_num*chunkread + ievt],batch[0]['trk_chi2rz'][batch_num*chunkread + ievt],res_func=linear_res_function(batch[0]['trk_pt'][batch_num*chunkread + ievt]), nbins=nbins, max_z0=max_z0)
        bendchi2_histo = CreateHisto(batch[0]['trk_z0'][batch_num*chunkread + ievt],batch[0]['trk_bendchi2'][batch_num*chunkread + ievt],res_func=linear_res_function(batch[0]['trk_pt'][batch_num*chunkread + ievt]), nbins=nbins, max_z0=max_z0)
        phi_histo = CreateHisto(batch[0]['trk_z0'][batch_num*chunkread + ievt],abs(batch[0]['trk_phi'][batch_num*chunkread + ievt]),res_func=linear_res_function(batch[0]['trk_pt'][batch_num*chunkread + ievt]), nbins=nbins, max_z0=max_z0)
        nstub_histo = CreateHisto(batch[0]['trk_z0'][batch_num*chunkread + ievt],batch[0]['trk_nstub'][batch_num*chunkread + ievt],res_func=linear_res_function(batch[0]['trk_pt'][batch_num*chunkread + ievt]), nbins=nbins, max_z0=max_z0)
        overeta_histo = CreateHisto(batch[0]['trk_z0'][batch_num*chunkread + ievt],trk_overEta,res_func=linear_res_function(batch[0]['trk_pt'][batch_num*chunkread + ievt]), nbins=nbins, max_z0=max_z0)

        trk_histo = CreateHisto(batch[0]['trk_z0'][batch_num*chunkread + ievt],ntrk,res_func=linear_res_function(batch[0]['trk_pt'][batch_num*chunkread + ievt]), nbins=nbins, max_z0=max_z0,factor=1)

        histo_list = [pt_histo[0],eta_histo[0],MVA_histo[0],chi2rphi_histo[0],chi2rz_histo[0],bendchi2_histo[0],phi_histo[0],nstub_histo[0],overeta_histo[0],trk_histo[0]]
        twod_hist = np.stack(histo_list, axis=0)

        twod_hist = np.nan_to_num(twod_hist)
        twod_hist /= np.max(twod_hist)
        Histograms[num_histos] = twod_hist

        Vertices[num_histos] = (batch[3]['pv_MC'][batch_num*chunkread + ievt][0])/max_z0
        num_histos += 1

        ##############################################################

    batch_num += 1

    #Add batch dataframe to larger dataframe, only defined branches are added to save memory
    print(batch_num," out of: ", len(events))

X_train, X_test, y_train, y_test = train_test_split(Histograms, Vertices, test_size=0.3, random_state=1)
X_train, X_val,  y_train, y_val  = train_test_split(X_train, y_train, test_size=0.5, random_state=1)

print("================= Train =================")
print(X_train.shape)
print(y_train.shape)
print("================= Validate =================")
print(X_val.shape)
print(y_val.shape)
print("================= Test =================")
print(X_test.shape)
print(y_test.shape)

# Save, can modify these locations as needed
os.system("mkdir "+name)
os.system("mkdir "+name+ "/Train")
os.system("mkdir "+name+ "/Test")
os.system("mkdir "+name+ "/Val")
os.system("mkdir "+name+ "/Plots")

np.save(name+"/Train/X.npy",X_train)
np.save(name+"/Val/X.npy",X_val)
np.save(name+"/Test/X.npy",X_test)

np.save(name+"/Train/y.npy",y_train)
np.save(name+"/Val/y.npy",y_val)
np.save(name+"/Test/y.npy",y_test)

plt.clf()
figure = plot_event(X_train[0],y_train[0]/max_z0,histo_names,max_z0,nbins)
plt.savefig("%s/%s.png" % (name+"/Plots", "train_histo"))
plt.close()

plt.clf()
figure = plot_event(X_val[0],y_val[0]/max_z0,histo_names,max_z0,nbins)
plt.savefig("%s/%s.png" % (name+"/Plots", "val_histo"))
plt.close()

plt.clf()
figure = plot_event(X_test[0],y_test[0]/max_z0,histo_names,max_z0,nbins)
plt.savefig("%s/%s.png" % (name+"/Plots", "test_histo"))
plt.close()
