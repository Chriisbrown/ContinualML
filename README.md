# ContinualAI

## Continual ML for Track to Vertex Association
## Description
This repo outlines the dataset generation, simple model definition and some evaluation functions for a track to vertex classifier. It acts as a starting point for the ML@L1 hackathon and will look to implement a continual ML model using the avalanche framework and pytorch


## Installation

With a standard install of a anaconda environment

`conda create env -f avalanche-environment.yml`

`source env.sh`

## Structure

`dataset/` contains the functiions for generating datasets from root files as well as default location for Train, Test and Val directories for training and evaluating

`model/` contains the model class definition, a simpleNN example is given. As well as default location for SavedModels where trained models are saved

`eval/` contains the eval_funcs used for evaluating trained models. As well as default location for plots where performance plots are saved

`continualML/` contains a simple avalanche example
## Usage
To generate datasets use in the dataset dir:

`dataset_generator.py path/to/rootfile.root`

To run a simple training of the simpleNN model use

`train.py`

## Support
Email c.brown19@imperial.ac.uk for questions

