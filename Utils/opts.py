import argparse
import Utils.opts_activitynet as opts_activitynet
import Utils.opts_thumos as opts_thumos
import Utils.opts_hacs as opts_hacs

def parse_opt():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='activitynet',
        type=str,
        choices=['thumos', 'activitynet', 'hacs'])

    parser.add_argument(
        '--iou_thr_bound',
        nargs='+',
        type=float,
        default=[0.45, 0.95]) # foregound:middle:background

    parser.add_argument(
        '--pretrain_model',
        default='none',
        type=str,
        choices=['none', 'FeatureEnhancer', 'detector', 'full'])

    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='checkpoint')
    parser.add_argument(
        '--output_path',
        type=str,
        default='./output')
    # "/media/zhaoc/Data/Datasets/Thumos14/thumos_feature_exp/TSN_pretrain_avepool_allfrms_hdf5/"
    # "/media/zhaoc/Data/Datasets/HACS"
    # "/media/zhaoc/Data/Datasets/ActivityNet1_3/TSN_features_rescaled1000/Anet_TSN_rf_rescaled1000_2.h5"
    parser.add_argument(
        '--feature_path',
        type=str,
        default="/media/zhaoc/Data/Datasets/Thumos14/thumos_feature_exp/TSN_pretrain_avepool_allfrms_hdf5/")
    parser.add_argument(
        '--is_train',
        default='true',
        type=str,
        choices=['true', 'false'])
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128)  #OG-->  default=256)

    # args = parser.parse_known_args(['dataset'])
    args = parser.parse_args()

    if args.dataset == 'thumos':
        opts_thumos.parse_opt(parser)
    elif  args.dataset == 'activitynet':
        opts_activitynet.parse_opt(parser)
    elif  args.dataset == 'hacs':
        opts_hacs.parse_opt(parser)

    args = parser.parse_args()

    return args