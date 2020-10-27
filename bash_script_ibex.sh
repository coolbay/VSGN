#!/bin/bash

#SBATCH --job-name sti_same
#SBATCH --array=1-6
#SBATCH --time=0-04:00:00
#SBATCH -o gpu.%A.out
#SBATCH -e gpu.%A.err

#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=25GB

set -ex
DATE_TIME=`date +'%Y-%m-%d_%H-%M-%S'`
DATASET='activitynet'

if [ ${DATASET} == 'thumos' ]
then
    FEAT_NAME='Thumos_TSN_org/'
elif [ ${DATASET} == 'activitynet' ]
then
    FEAT_NAME='Anet_TSN_org'
elif [ ${DATASET} == 'hacs' ]
then
    FEAT_NAME='HACS/'
fi
echo $SLURM_ARRAY_TASK_ID
IOU_BOUND='0.45 0.95'
TRAIN_LR=0.001
GAP=$(sed -n "$((SLURM_ARRAY_TASK_ID))"p gap.txt)

TRAIN_FLAG="${DATASET}_${DATE_TIME}_lr${TRAIN_LR}_gap${GAP}"
CKP_PATH=./checkpoint_${TRAIN_FLAG}
OUTPUT_PATH=./output_${TRAIN_FLAG}
LOG_TRAIN="${CKP_PATH}/log_train.txt"
LOG_TEST="${OUTPUT_PATH}/log_test.txt"

# Choose machine
if  [ $1 == 'kw60749' ]
then
    DATA_PATH="/home/xum/dataset/${FEAT_NAME}"
    module load cuda
    source activate pytorch110
elif [ $1 == 'kw60748' ] || [ $1 == 'kw60747' ] || [ $1 == 'kw60746' ] || [ $1 == 'kw60623' ] ||  [ $1 == 'kw60661' ] || [ $1 == 'kw60624' ]
then
    DATA_PATH="/home/zhaoc/datasets/${FEAT_NAME}"
    source activate pytorch110
elif [ $1 == 'ibex' ]
then
    DATA_PATH="/ibex/scratch/zhaoc/datasets/${FEAT_NAME}"
    OUT_PMAP='false'
    module purge
    module load anaconda3/2019.10
    module load cuda/10.0.130
    source activate pytorch110
fi

# Choose train or infer
if [[ $2 =~ .*'train'.* ]]
then
    if [ $3 ]
    then
        TRAIN_FLAG=$3
        CKP_PATH=./checkpoint_${TRAIN_FLAG}
    else
        mkdir -p ${CKP_PATH}
    fi
    echo Logging output to "$LOG_TRAIN"
    python Train.py  --iou_thr_bound ${IOU_BOUND} \
        --feature_path ${DATA_PATH} \
        --checkpoint_path ${CKP_PATH}  \
        --is_train true   \
        --dataset ${DATASET}   \
        --batch_size  256  \
	    --train_lr ${TRAIN_LR}  \
	    --stitch_gap ${GAP} | tee -a "$LOG_TRAIN"
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
        --batch_size  256  \
	    --stitch_gap ${GAP}  | tee -a "$LOG_TEST"
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
        --dataset ${DATASET}  \
        --out_prop_map 'false'  | tee -a "$LOG_TEST"
fi

conda deactivate