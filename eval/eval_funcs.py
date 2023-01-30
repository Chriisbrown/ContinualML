import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
#hep.set_style("CMSTex")

import sklearn.metrics as metrics
from textwrap import wrap


"""
These function define evaluation plots to be run on the output of the model

"""

# Setup plotting to CMS style
hep.cms.label()
hep.cms.text("Simulation")
plt.style.use(hep.style.CMS)

SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 30

LEGEND_WIDTH = 20
LINEWIDTH = 3
MARKERSIZE = 20

colormap = "jet"

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('axes', linewidth=LINEWIDTH+2)              # thickness of axes
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE-2)            # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

matplotlib.rcParams['xtick.major.size'] = 20
matplotlib.rcParams['xtick.major.width'] = 5
matplotlib.rcParams['xtick.minor.size'] = 10
matplotlib.rcParams['xtick.minor.width'] = 4

matplotlib.rcParams['ytick.major.size'] = 20
matplotlib.rcParams['ytick.major.width'] = 5
matplotlib.rcParams['ytick.minor.size'] = 10
matplotlib.rcParams['ytick.minor.width'] = 4

colours=["red","green","blue","orange","purple","yellow"]

def plot_event(twod_histo,y,feature_names,max_z0,nbins):
    fig,ax = plt.subplots(1,1,figsize=(24,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    
    nan_array = np.zeros_like(twod_histo[0])
    nan_array[:] = np.NaN
    nan_array = np.expand_dims(nan_array,axis=0)
    twod_histo = np.vstack([twod_histo,nan_array])
    hist2d = ax.imshow(twod_histo,cmap=colormap,aspect='auto',extent=[-1*max_z0,max_z0,0,len(feature_names)])

    ax.grid(True,axis='y',linewidth=2)
    ax.grid(True,axis='x',linewidth=1)
    ax.set_ylabel('Track Feature',ha="right",y=1)
    ax.set_xlabel('Track $z_{0}$ [cm]',ha="right",x=1)
        
    ax.set_yticklabels(feature_names)
    ax.set_yticks(np.array([1,2,3,4,5,6,7,8,9,10,11]))

    rect = plt.Rectangle((y-((2.5*max_z0)/nbins), 1), 5*max_z0/nbins, len(feature_names),
                                    fill=False,linewidth=2,linestyle='--',edgecolor='r')
    ax.add_patch(rect)
    ax.text(y-0.5, 0.5, "True Vertex", color='r')

    cbar = plt.colorbar(hist2d , ax=ax)
    cbar.set_label('Weighted Density')

    cbar.set_label('Weighted Density')
    ax.tick_params(axis='y', which='minor', right=False,left=False)
    plt.tight_layout()
    return fig


def plotz0_residual(actual,predicted,names,title="None",max_z0=20.46912512,colours=colours,):
    plt.clf()
    fig,ax = plt.subplots(1,2,figsize=(20,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[0])
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[1])
    
    print(len(predicted))
    items = 0
    for i,prediction in enumerate(predicted):
        print(len(prediction))
        FH = actual - prediction
        qz0_FH = np.percentile(FH,[32,50,68])
        ax[0].hist(FH,bins=50,range=(-1*max_z0,max_z0),histtype="step",
                 linewidth=LINEWIDTH,color = colours[items],
                 label='\n'.join(wrap(f"%s \nRMS = %.4f" 
                 %(names[i],np.sqrt(np.mean(FH**2))),LEGEND_WIDTH)),density=True)
        ax[1].hist(FH,bins=50,range=(-1,1),histtype="step",
                 linewidth=LINEWIDTH,color = colours[items],
                 label='\n'.join(wrap(f"%s \nQuartile Width = %.4f" 
                 %(names[i],qz0_FH[2]-qz0_FH[0]),LEGEND_WIDTH)),density=True)
        items+=1
    
    ax[0].grid(True)
    ax[0].set_xlabel('$z^{PV}_0$ Residual [cm]',ha="right",x=1)
    ax[0].set_ylabel('Events',ha="right",y=1)
    ax[0].set_yscale("log")
    ax[0].legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))

    ax[1].grid(True)
    ax[1].set_xlabel('$z^{PV}_0$ Residual [cm]',ha="right",x=1)
    ax[1].set_ylabel('Events',ha="right",y=1)
    ax[1].legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))

    plt.suptitle(title)
    plt.tight_layout()
    return fig

def predictFastHisto(twod_histo, max_z0=20.46912512,nbins=256):
    z0List = []
    halfBinWidth = 0.5*(2*max_z0)/nbins

    for ibatch in range(twod_histo.shape[0]):
        hist = np.convolve(twod_histo[ibatch][0][0],[1,1,1],mode='same')
        z0Index= np.argmax(hist)
        z0 = -1*max_z0 +(2*max_z0)*z0Index/nbins + halfBinWidth
        z0List.append([z0])
    return np.array(z0List,dtype=np.float32)