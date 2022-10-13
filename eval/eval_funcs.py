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

def plotPV_roc(actual,predictions,names,Nthresholds=50,colours=colours):
    '''
    Plots reciever operating characteristic curve for output of a predicition model

    Takes: 
        actual: a numpy array of true values 0 or 1
        predictions: a list of numpy arrays, each array same length as actual which are probabilities of coming from class 1, float between 0 and 1
        names: a list of strings naming each of the prediciton arrays
        Nthresholds: how many thresholds between 0 and 1 to calcuate the TPR, FPR etc.
        colours: list of matplotlib colours to be used for each item in the predictions list

    '''
    fig,ax = plt.subplots(1,2,figsize=(20,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[0])
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[1])

    items = 0

    # Iterate through predictions
    for i,prediction in enumerate(predictions):

        precision = []
        recall = []
        FPR= []

        thresholds = np.linspace(0,1,Nthresholds)

        #Iterate through thresholds
        for j,threshold in enumerate(thresholds):
            print(str(names[i]) + " Testing ROC threshold: "+str(j) + " out of "+str(len(thresholds)))
            # Find number of true negatives, false positive, false negatives and true positives when decision bounary == threshold
            tn, fp, fn, tp = metrics.confusion_matrix(actual, prediction>threshold).ravel()
            precision.append( tp / (tp + fp) )
            recall.append(tp / (tp + fn) )
            FPR.append(fp / (fp + tn) )

        # Plot precision recall and ROC curves
        ax[0].plot(recall,precision,label=str(names[i]),linewidth=LINEWIDTH,color=colours[items])
        ax[1].plot(recall,FPR,linewidth=LINEWIDTH,label='\n'.join(wrap(f"%s AUC: %.4f" %(names[i],metrics.roc_auc_score(actual,prediction)),LEGEND_WIDTH)),color=colours[items])
        items += 1

    ax[0].grid(True)
    ax[0].set_xlabel('Efficiency',ha="right",x=1)
    ax[0].set_ylabel('Purity',ha="right",y=1)
    ax[0].set_xlim([0,0.75])
    ax[0].legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))

    ax[1].grid(True)
    #ax[1].set_yscale("log")
    ax[1].set_xlabel('Track to Vertex Association True Positive Rate',ha="right",x=1)
    ax[1].set_ylabel('Track to Vertex Association False Positive Rate',ha="right",y=1)
    ax[1].set_xlim([0.75,1])
    ax[1].set_ylim([1e-2,1])
    ax[1].legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))
    plt.tight_layout()
    return fig


def plot_split_histo(actual,variable,variable_name,range=(0,1),bins=100):
    pv_track_sel = actual == 1
    pu_track_sel = actual == 0
    fig,ax = plt.subplots(1,1,figsize=(12,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax)
    
    ax.hist(variable[pv_track_sel],range=range, bins=bins, label="PV tracks", density=True,histtype="stepfilled",color='g',alpha=0.7,linewidth=LINEWIDTH)
    ax.hist(variable[pu_track_sel],range=range, bins=bins, label="PU tracks", density=True,histtype="stepfilled",color='r',alpha=0.7,linewidth=LINEWIDTH)
    ax.set_xlabel(variable_name, horizontalalignment='right', x=1.0)
    ax.set_ylabel("Fraction of Tracks", horizontalalignment='right', y=1.0)
    ax.set_yscale("log")
    ax.legend()
    ax.tick_params(axis='x', which='minor', bottom=False,top=False)
    plt.tight_layout()

    return fig
