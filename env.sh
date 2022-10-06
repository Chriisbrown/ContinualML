#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate avalanche-dev-env

pip install avalanche-lib

export PATH=~/anaconda3/envs/avalanche-dev-env/bin/python:$PATH
export PYTHONPATH=$PWD:$PYTHONPATH 

mkdir dataset/Test
mkdir dataset/Train
mkdir dataset/Val