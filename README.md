# ContinualAI

## Continual ML for Track to Vertex Association
## Description
This repo outlines the dataset generation, simple model definition and some evaluation functions for a track to vertex classifier. It acts as a starting point for the ML@L1 hackathon and will look to implement a continual ML model using the avalanche framework and pytorch

Some dataset exist here: https://cernbox.cern.ch/index.php/s/9P2Qw1ssGcld3Pz

OldKF_test.root 16K events

GTT_TrackNtuple_FH_oldTQ.root 300K events 


## Installation

With a standard install of a anaconda environment

`conda env create -f avalanche-environment.yml`

`source env.sh`

To install miniconda to setup an anaconda environment

`wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`

`bash Miniconda3-latest-Linux-x86_64.sh`

`conda env create -f avalanche-environment.yml`


## Structure

`dataset/` contains the functiions for generating datasets from root files as well as default location for Train, Test and Val directories for training and evaluating

`model/` contains the model class definition, a simpleNN example is given. As well as default location for SavedModels where trained models are saved

`eval/` contains the eval_funcs used for evaluating trained models. As well as default location for plots where performance plots are saved

`continualML/` contains a simple avalanche example
## Usage
To generate datasets use in the dataset dir:

`python dataset_generator.py path/to/rootfile.root`

For large datasets this will take a while 

To run a simple training of the simpleNN model use

`python train.py`


To evaluate a model use

`python eval.py`
## Support
Email c.brown19@imperial.ac.uk for questions

