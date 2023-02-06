
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/video-self-stitching-graph-network-for/temporal-action-localization-on-thumos14)](https://paperswithcode.com/sota/temporal-action-localization-on-thumos14?p=video-self-stitching-graph-network-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/video-self-stitching-graph-network-for/temporal-action-localization-on-activitynet)](https://paperswithcode.com/sota/temporal-action-localization-on-activitynet?p=video-self-stitching-graph-network-for)
# VSGN

This repo holds the codes of paper: "[Video Self-Stitching Graph Network for Temporal Action Localization](https://arxiv.org/abs/2011.14598)", accepted to ICCV 2021.

## Updates
 - Aug. 15th, 2021: Code and pre-trained model on THUMOS14 are released.
 - Feb. 6th, 2023: Uncleaned code on ActivityNet TSN features is released (not verified, there might be many issues).

## VSGN Introduction
![VSGN Overview](./VSGN_overview.png)
Temporal action localization (TAL) in videos is a challenging task, especially due to the large variation in action temporal scales. Short actions usually occupy a major proportion in the datasets, but tend to have the lowest performance. In this paper, we confront the challenge of short actions and propose a multi-level cross-scale solution dubbed as video self-stitching graph network (VSGN). We have two key components in VSGN: video self-stitching (VSS) and cross-scale graph pyramid network (xGPN). In VSS, we focus on a short period of a video and magnify it along the temporal dimension to obtain a larger scale. We stitch the original clip and its magnified counterpart in one input sequence to take advantage of the complementary properties of both scales. The xGPN component further exploits the cross-scale correlations by a pyramid of cross-scale graph networks, each containing a hybrid module to aggregate features from across scales as well as within the same scale. Our VSGN not only enhances the feature representations, but also generates more positive anchors for short actions and more short training samples. Experiments demonstrate that VSGN obviously improves the localization performance of short actions as well as achieving the state-of-the-art overall performance on THUMOS-14 and ActivityNet-v1.3.

## Project Architecture
An overview of the project architecture in repo is shown below.
```
    VSGN                            
    ├── Models/*                    # Network modules and losses
    ├── Utils/*                     # Data loading and hyper-parameters
    ├── Evaluation/*                # Post-processing and performance evaluation
    ├── DETAD/*                     # DETAD evaluation to generate performance for different action duration   
    ├── Cut_long_videos.py          # Cutting long videos      
    ├── Train.py                    # Training starts from here      
    ├── Infer.py                    # Inference starts from here    
    ├── Eval.py                     # Evaluation starts from here             
    └── ...
```
- Functions for video cutting in **VSS** are in [`Cut_long_videos.py`](./Cut_long_videos.py).
- Functions for clip up-scaling and self-stitching in **VSS** are in [`Utils/dataset_activitynet.py`](./Utils/dataset_activitynet.py).
- Network model is defined in [`Model/VSGN.py`](./Models/VSGN.py), with detailed implementation of of different modules in different files in [`Models`](./Models).
- Losses are defined in [`Models/Loss.py`](./Models/Loss.py)
- We use pre-extracted video features. ActivityNet features can be found: [TSN](https://drive.google.com/drive/folders/1X9V7eTFmkfzSZJqCPnJu7x3xP9Xg428Z?usp=share_link), [I3D](https://drive.google.com/drive/folders/1S3Ub0snH2-71OgGViIGdgHYqdRmsKrYV?usp=share_link).




## Environment Installation
Create a conda environment and install required packages from scratch following the steps below
```
    conda create -n pytorch160 python=3.7 
    conda activate pytorch160   
    conda install pytorch=1.6.0 torchvision cudatoolkit=10.1.243 -c pytorch   
    conda install -c anaconda pandas    
    conda install -c anaconda h5py  
    conda install -c anaconda scipy 
    conda install -c conda-forge tensorboardx   
    conda install -c anaconda joblib    
    conda install -c conda-forge matplotlib 
    conda install -c conda-forge urllib3
```
OR you can create a conda environment from our `env.yml` file using the following command
```
    conda env create -f env.yml
```


## Code and Data Preparation
Download the TSN features of the ActivityNet dataset.

Clone this repo with git
```
    git clone git@github.com:coolbay/VSGN.git
```

## Run the Code

### Prepare input by cutting videos
```
    python Cut_long_videos.py 
```


### Training

```    
     python Train.py --is_train true --dataset activitynet --feature_path DATA_PATH  --checkpoint_path CHECKPOINT_PATH  
```
### Inference
```
     python Infer.py --is_train false --dataset activitynet --feature_path DATA_PATH --checkpoint_path CHECKPOINT_PATH  --output_path OUTPUT_PATH   
```
### Evaluation
```
     python Eval.py --dataset activitynet --output_path [OUTPUT_PATH]
```
### Run training / inference / evaluation in one command
```
    bash run_vsgn.sh traininfereval     # Run train, infer, and eval 
    bash run_vsgn.sh train              # Only run train
    bash run_vsgn.sh infer              # Only run infer
    bash run_vsgn.sh eval               # Only run eval
    bash run_vsgn.sh traininfer         # Run train and infer
```
## Cite this paper
Please cite the following paper if this codebase is useful for your work.
```
    @article{zhao2020video,
      title={Video Self-Stitching Graph Network for Temporal Action Localization},
      author={Zhao, Chen and Thabet, Ali and Ghanem, Bernard},
      journal={arXiv preprint arXiv:2011.14598},
      year={2020}
    }
```
## Acknowledgements

VSGN is built by referring to the implementation of [G-TAD](https://arxiv.org/pdf/1911.11462.pdf), [BSN](https://arxiv.org/pdf/1806.02964.pdf), [ATSS](https://arxiv.org/pdf/1912.02424.pdf) and the description in [PBRNet]((https://ojs.aaai.org/index.php/AAAI/article/view/6829)). If you use our model, please consider citing these works as well.

- G-TAD: https://github.com/frostinassiky/gtad
- DETAD: https://github.com/HumamAlwassel/DETAD
- BSN: https://github.com/wzmsltw/BSN-boundary-sensitive-network.pytorch
- ATSS: https://github.com/sfzhang15/ATSS
- PBRNet: Qinying Liu, Zilei Wang, [Progressive Boundary Refinement Network for Temporal Action Detection](https://ojs.aaai.org/index.php/AAAI/article/view/6829), AAAI'20.




