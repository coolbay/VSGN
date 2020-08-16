#!/usr/bin/env bash

    conda create -n pytorch110 python=3.7 &&
    conda activate pytorch110   &&
    conda install pytorch=1.1.0 torchvision cudatoolkit=10.0.130 -c pytorch   &&
    conda install -c anaconda pandas    &&
    conda install -c anaconda h5py  &&
    conda install -c anaconda scipy &&
    conda install -c conda-forge tensorboardx   &&
    conda install -c anaconda joblib    &&
    conda install -c conda-forge matplotlib &&
    conda install -c conda-forge urllib3