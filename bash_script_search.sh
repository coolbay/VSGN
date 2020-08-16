#!/bin/bash

#SBATCH --job-name stl
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:v100:4
#SBATCH -o gpu.%A.out
#SBATCH -e gpu.%A.err
#SBATCH --cpus-per-task 20
#SBATCH --mem 100GB

set -ex

DATASET='activitynet'

if [ ${DATASET} == 'thumos' ]
then
    FEAT_NAME='Thumos_TSN_org/'
elif [ ${DATASET} == 'activitynet' ]
then
    FEAT_NAME='TSN_features_rescaled1000/Anet_TSN_rf_rescaled1000_2.h5'
elif [ ${DATASET} == 'hacs' ]
then
    FEAT_NAME='HACS/'
fi


# Choose machine
if  [ $1 == 'kw60749' ]
then
    DATA_PATH="/home/xum/dataset/${FEAT_NAME}"
    module load cuda
    source activate pytorch110
elif [ $1 == 'kw60748' ] || [ $1 == 'kw60747' ] || [ $1 == 'kw60746' ] || [ $1 == 'kw60623' ]
then
    DATA_PATH="/home/zhaoc/datasets/${FEAT_NAME}"
    source activate pytorch110
elif [ $1 == 'ibex' ]
then
    DATA_PATH="/ibex/scratch/zhaoc/datasets/${FEAT_NAME}"
    OUT_PMAP='false'
    module purge
    module load anaconda3
    module load cuda
    source activate pytorch110
fi

for low in 0.25 0.35 0.45 0.5 0.55 0.65 0.7 0.75 0.85 0.9 0.95; do
    for high in 0.5 0.55 0.65 0.7 0.75 0.85 0.9 0.95; do
        IOU_BOUND="${low} ${high}"

        DATE_TIME=`date +'%Y-%m-%d_%H-%M'`
        TRAIN_FLAG="${DATASET}_${DATE_TIME}_${low}_${high}"
        CKP_PATH=./checkpoint_${TRAIN_FLAG}
        OUTPUT_PATH=./output_${TRAIN_FLAG}
        LOG_TRAIN="${CKP_PATH}/log_train.txt"
        LOG_TEST="${OUTPUT_PATH}/log_test.txt"


        # Choose train or infer
        if [[ $2 =~ .*'train'.* ]]
        then
            mkdir -p ${CKP_PATH}
            echo Logging output to "$LOG_TRAIN"
            python Train.py  --iou_thr_bound ${IOU_BOUND} \
                --feature_path ${DATA_PATH} \
                --checkpoint_path ${CKP_PATH}  \
                --is_train true   \
                --dataset ${DATASET}   \
                --batch_size  128  | tee -a "$LOG_TRAIN"
        fi

        if [[ $2 =~ .*'infer'.* ]]
        then
            if [ $3 ]
            then
                TRAIN_FLAG=$3
                CKP_PATH=./checkpoint_${TRAIN_FLAG}
                OUTPUT_PATH=./output_${TRAIN_FLAG}
                LOG_TEST="${OUTPUT_PATH}/log_test.txt"
            fi
            mkdir -p ${OUTPUT_PATH}
            echo Logging output to "$LOG_TEST"
            python Infer.py  --output_path ${OUTPUT_PATH}    \
                --feature_path ${DATA_PATH} \
                --checkpoint_path ${CKP_PATH}   \
                --is_train false  \
                --dataset ${DATASET}   \
                --batch_size  8  | tee -a "$LOG_TEST"
        fi

        if [[ $2 =~ .*'eval'.* ]]
        then
            if [ $3 ]
            then
                TRAIN_FLAG=$3
                CKP_PATH=./checkpoint_${TRAIN_FLAG}
                OUTPUT_PATH=./output_${TRAIN_FLAG}
                LOG_TEST="${OUTPUT_PATH}/log_test.txt"
            fi
            echo Logging output to "$LOG_TEST"
            python Eval.py  --output_path ${OUTPUT_PATH}    \
                --feature_path ${DATA_PATH} \
                --checkpoint_path ${CKP_PATH}   \
                --dataset ${DATASET}   | tee -a "$LOG_TEST"
        fi
    done
done

conda deactivate