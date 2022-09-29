import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
#hep.set_style("CMSTex")

import sklearn.metrics as metrics
from textwrap import wrap

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
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('axes', linewidth=5)              # thickness of axes
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)            # legend fontsize
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

def plotPV_roc(actual,pred,name,Nthresholds=50,colours=colours):
    plt.clf()
    fig,ax = plt.subplots(1,2,figsize=(20,10))
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[0])
    hep.cms.label(llabel="Phase-2 Simulation Preliminary",rlabel="14 TeV, 200 PU",ax=ax[1])

    items=0

    precisionNN = []
    recallNN = []
    FPRNN = []

    thresholds = np.linspace(0,1,Nthresholds)

    for j,threshold in enumerate(thresholds):
        print(str(name) + " Testing ROC threshold: "+str(j) + " out of "+str(len(thresholds)))
        tnNN, fpNN, fnNN, tpNN = metrics.confusion_matrix(actual, pred>threshold).ravel()
        precisionNN.append( tpNN / (tpNN + fpNN) )
        recallNN.append(tpNN / (tpNN + fnNN) )
        FPRNN.append(fpNN / (fpNN + tnNN) )

    ax[0].plot(recallNN,precisionNN,label=str(name),linewidth=LINEWIDTH,color=colours[items])
    ax[1].plot(recallNN,FPRNN,linewidth=LINEWIDTH,label='\n'.join(wrap(f"%s AUC: %.4f" %(name,metrics.roc_auc_score(actual,pred)),LEGEND_WIDTH)),color=colours[items])

    ax[0].grid(True)
    ax[0].set_xlabel('Efficiency',ha="right",x=1)
    ax[0].set_ylabel('Purity',ha="right",y=1)
    ax[0].set_xlim([0,0.75])
    ax[0].legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))

    ax[1].grid(True)
    ax[1].set_yscale("log")
    ax[1].set_xlabel('Track to Vertex Association True Positive Rate',ha="right",x=1)
    ax[1].set_ylabel('Track to Vertex Association False Positive Rate',ha="right",y=1)
    ax[1].set_xlim([0.75,1])
    ax[1].set_ylim([1e-2,1])
    ax[1].legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))
    plt.tight_layout()
    return fig