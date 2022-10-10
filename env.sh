#!/bin/bash

#If your shell does not automatically activate conda (will need adapting to your anaconda install)
#source ~/anaconda3/etc/profile.d/conda.sh
conda activate avalanche-dev-env

avalancheVAR=$(pip list | grep avalanche-lib)
if [ -z "$avalancheVAR" ]
then
    pip install avalanche-lib
fi

mplhepVAR=$(pip list | grep mplhep)
if [ -z "$mplhepVAR" ]
then
    pip install mplhep
fi

# If your shell does not automatically activate conda (will need adapting to your anaconda install)
#export PATH=~/anaconda3/envs/avalanche-dev-env/bin/python:$PATH
export PYTHONPATH=$PWD:$PYTHONPATH 

mkdir dataset/Test
mkdir dataset/Train
mkdir dataset/Val
