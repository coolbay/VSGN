#!/bin/bash


DATASET='thumos'
DATE_TIME=`date +'%Y-%m-%d_%H-%M-%S'`
TRAIN_FLAG="${DATASET}_${DATE_TIME}"
CKP_PATH=./checkpoint_${TRAIN_FLAG}
OUTPUT_PATH=./output_${TRAIN_FLAG}

DATA_PATH="/mnt/sdb1/Datasets/Thumos14/thumos_feature_exp/TSN_pretrain_avepool_allfrms_hdf5"
LOG_TRAIN="${CKP_PATH}/log_train.txt"
LOG_TEST="${OUTPUT_PATH}/log_test.txt"

source activate pytorch110

# Choose train or infer
if [[ $1 =~ .*'train'.* ]]
then
    if [ $2 ]
    then
        TRAIN_FLAG=$2
        CKP_PATH=./checkpoint_${TRAIN_FLAG}
    else
        mkdir -p ${CKP_PATH}
    fi
    echo Logging output to "$LOG_TRAIN"
    python Train.py \
        --is_train true   \
        --dataset ${DATASET}   \
        --feature_path ${DATA_PATH} \
        --checkpoint_path ${CKP_PATH}  | tee -a "$LOG_TRAIN"
fi

if [[ $1 =~ .*'infer'.* ]]
then
    if [ $2 ]
    then
        TRAIN_FLAG=$2
        CKP_PATH=./checkpoint_${TRAIN_FLAG}
        OUTPUT_PATH=./output_${TRAIN_FLAG}
        LOG_TEST="${OUTPUT_PATH}/log_test.txt"
    fi
    mkdir -p ${OUTPUT_PATH}
    echo Logging output to "$LOG_TEST"
    python Infer.py   \
        --is_train false  \
        --dataset ${DATASET}   \
        --feature_path ${DATA_PATH} \
        --checkpoint_path ${CKP_PATH}  \
        --output_path ${OUTPUT_PATH}    | tee -a "$LOG_TEST"
fi

if [[ $1 =~ .*'eval'.* ]]
then
    if [ $2 ]
    then
        TRAIN_FLAG=$2
        CKP_PATH=./checkpoint_${TRAIN_FLAG}
        OUTPUT_PATH=./output_${TRAIN_FLAG}
        LOG_TEST="${OUTPUT_PATH}/log_test.txt"
    fi
    echo Logging output to "$LOG_TEST"
    python Eval.py  \
      --dataset ${DATASET}  \
      --output_path ${OUTPUT_PATH}   | tee -a "$LOG_TEST"
fi

conda deactivate
