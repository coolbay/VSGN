import argparse

def parse_opt():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default='thumos',
        type=str,
        choices=['thumos', 'activitynet'])
    parser.add_argument(
        '--is_train',
        default='true',
        type=str,
        choices=['true', 'false'])
    parser.add_argument(
        '--out_prop_map',
        default='true',
        type=str,
        choices=['true', 'false'])

    # Dataset and annotation paths
    parser.add_argument(
        '--feature_path',
        type=str,
        default="/mnt/sdb1/Datasets/Thumos14/thumos_feature_exp/TSN_pretrain_avepool_allfrms_hdf5/")
    parser.add_argument(
        '--video_info',
        type=str,
        default="Evaluation/thumos/annot/thumos14_test_groundtruth.csv")
    parser.add_argument(
        '--video_anno',
        type=str,
        default="Evaluation/thumos/annot/thumos_gt.json")
    parser.add_argument(
        '--thumos_classes',
        type=str,
        default="Evaluation/thumos/annot/thumos_classes_idx.json")

    # Output paths
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='checkpoint')
    parser.add_argument(
        '--output_path',
        type=str,
        default='output')
    parser.add_argument(
        '--prop_path',
        type=str,
        default='proposals')
    parser.add_argument(
        '--prop_result_file',
        type=str,
        default="proposals_postNMS.json")
    parser.add_argument(
        '--detect_result_file',
        type=str,
        default="detections_postNMS.json")
    parser.add_argument(
        '--detad_sensitivity_file',
        type=str,
        default="detad_sensitivity")

    # Training hyper-parameters
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128)
    parser.add_argument(
        '--train_lr',
        type=float,
        default=0.00005)
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0001)
    parser.add_argument(
        '--num_epoch',
        type=int,
        default=5)  # org 30
    parser.add_argument(
        '--step_size',
        type=int,
        default=15)
    parser.add_argument(
        '--step_gamma',
        type=float,
        default=0.1)
    parser.add_argument(
        '--focal_alpha',
        type=float,
        default=0.01)

    # Post processing
    parser.add_argument(
        '--nms_alpha_detect',
        type=float,
        default=0.46)
    parser.add_argument(
        '--nms_alpha_prop',
        type=float,
        default=0.75)
    parser.add_argument(
        '--nms_thr',
        type=float,
        default=0.4)

    # Model architecture settings
    parser.add_argument(
        '--temporal_scale',
        type=int,
        default=1280)
    parser.add_argument(
        '--input_feat_dim',
        type=int,
        default=2048)
    parser.add_argument(
        '--bb_hidden_dim',
        type=int,
        default=256)
    parser.add_argument(
        '--decoder_num_classes',
        type=int,
        default=21)
    parser.add_argument(
        '--num_levels',
        type=int,
        default=5)  # 5
    parser.add_argument(
        '--num_head_layers',
        type=int,
        default=4)

    # Graph network hyper-parameters
    parser.add_argument(
        '--nfeat_mode',
        default='feat_ctr',
        type=str,
        choices=['feat_ctr', 'dif_ctr', 'feat'])
    parser.add_argument(
        '--num_neigh',
        type=int,
        default=10)
    parser.add_argument(
        '--edge_weight',
        default='false',
        type=str,
        choices=['true', 'false'])
    parser.add_argument(
        '--agg_type',
        default='max',
        type=str,
        choices=['max', 'mean'])
    parser.add_argument(
        '--gcn_insert',
        default='par',
        type=str,
        choices=['seq', 'par'])

    # Detection hyper-parameters
    parser.add_argument(
        '--iou_thr',
        nargs='+',
        type=float,
        default=[0.5, 0.5, 0.7])
    parser.add_argument(
        '--anchor_scale',
        nargs='+',
        type=float,
        default=[1, 1.5])  # 4, 6; 8, 12; 16, 24; 32, 48; 64, 96
    parser.add_argument(
        '--base_stride',
        type=int,
        default=4)


    # VSS hyper-parameters
    parser.add_argument(
        '--stitch_gap',
        type=int,
        default=30)
    parser.add_argument(
        '--short_ratio',
        type=float,
        default=0.4)
    parser.add_argument(
        '--clip_win_size',
        type=float,
        default=0.38)

    args = parser.parse_args()

    return args
