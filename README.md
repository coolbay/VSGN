
# Deeplab multi-label 

## Ref
- **Actionness:** BSN: Boundary Sensitive Network for Temporal Action Proposal Generation [https://github.com/wzmsltw/BSN-boundary-sensitive-network.pytorch]
- **Deeplab:** Rethinking Atrous Convolution for Semantic Image Segmentation [https://github.com/jfzhang95/pytorch-deeplab-xception/blob/9135e104a7a51ea9effa9c6676a2fcffe6a6a2e6/modeling/deeplab.py]
- **Boundary map:** BMN: Boundary-Matching Network for Temporal Action Proposal Generation [https://github.com/JJBOY/BMN-Boundary-Matching-Network] 
- **TAG:** Temporal Action Detection with Structured Segment Networks [https://github.com/yjxiong/action-detection]
- **Adaptive IoU threshold**: Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection [https://github.com/sfzhang15/ATSS]


# Prerequisites and Installation

These code is  implemented in Pytorch 1.4.0 + Python3.6 + tensorboardX. Thus please install Pytorch first.

1. Create conda environment from the yml file

    ```
        conda env create -f env.yml 
    ```
    Or create an empty environment from scracth and install all the libraries specified in the file `install.sh` manually.
2. Compile the CUDA code Align1D
    ```
        source activate pytorch110
        cd Utils/RoIAlign
        python setup.py install
    ```

# Code and Data Preparation



# Training and Testing  

All configurations are saved in opts.py, where you can modify training and model parameter.

```
    bash bash_script.sh [datacenter/zhaocws/ibex] [train/infer/traninfer]
```


### Coding issues and solutions
1. Fail to use Align1D.
 
- I sucessfully installed the C++ library of Align1D, but have the following error when running: 
```
    undefined symbol: cudaSetupArgument
```

- After I switch my CUDA to the specified 10 version, I don't have the first error anymore, but have anonther one as follows.
```
    runtimeError: cuda runtime error (98) : invalid device function
```

This is because I installed my pytorch using 
```
    conda install pytorch=1.4.0 torchvision=0.4.0 cudatoolkit=10.3 -c pytorch 
```
but the cuda version on my workstation is actually 10.0. So I downgrade the cudatoolkit by the following
```
    conda install pytorch=1.4.0 torchvision cudatoolkit=10.0 -c pytorch 
```
Ref: https://github.com/open-mmlab/mmdetection/issues/1705#issuecomment-557762700

Ref: https://github.com/facebookresearch/detectron2/issues/62#issuecomment-542447753

- On IBEX, I have the following error when running
```
    RuntimeError: cuda runtime error (209) : no kernel image is available for execution on the device at Align1D_cuda_kernal.cu:239
```

2. Infer: when running the following code on datacenter
 ```
    pred_bdmap_batch = pred_bd_map.detach().cpu().numpy()  
 ```
    comes this error:
 ```
     CUDA ERROR: an illegal memory
 ```
 Solution: change pytorch env to the 1.1.0 one
 
 
 
 ##### Possible reasons for performance discrepency between original 125 implementation and point selecting using alter_window:
- No learning rate scheduling * 
- One less conv layer * 
- Boundary extention is one frame more
- Inference 'end' is one frame different
 

test_info.mat
val_info.mat
thumos_classes_idx.json

3. thumos14_test_groundtruth.csv
5. thumos_gt.json
6. test_Annotation.csv
