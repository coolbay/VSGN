import argparse

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

    parser.add_argument(
        '--feature_path',
        type=str,
        default="/media/zhaoc/Data/Datasets/ActivityNet1_3/TSN_features_rescaled1000/Anet_TSN_rf_rescaled1000_2.h5")
    parser.add_argument(
        '--is_train',
        default='true',
        type=str,
        choices=['true', 'false'])
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128)  #OG-->  default=256)

    # Dataset paths settings
    parser.add_argument(
        '--video_info',
        type=str,
        default="Evaluation/activitynet/annot/video_info_new.csv")
    parser.add_argument(
        '--video_anno',
        type=str,
        default='Evaluation/activitynet/annot/anet_anno_action.json') #  Evaluation/activitynet/annot/anet_anno_action.json
    parser.add_argument(
        '--detad_anno',
        type=str,
        default="Evaluation/activitynet/annot/activity_net_train_val_extra_characteristics.v1-3.min.json")
    parser.add_argument(
        '--eval_anno',
        type=str,
        default="Evaluation/activitynet/annot/activity_net_1_3_new.json")
    parser.add_argument(
        '--vlevel_cls_res',
        type=str,
        default="Evaluation/activitynet/annot/cuhk_val_simp_share.json")
    parser.add_argument(
        '--anet_classes',
        type=str,
        default="Evaluation/activitynet/annot/anet_classes_idx.json")

    # Output paths settings

    parser.add_argument(
        '--prop_path',
        type=str,
        default='proposals')
    parser.add_argument(
        '--actionness_path',
        type=str,
        default='actionness')
    parser.add_argument(
        '--start_end_path',
        type=str,
        default='start_end')
    parser.add_argument(
        '--prop_map_path',
        type=str,
        default='prop_map')
    parser.add_argument(
        '--action_loss_file',
        type=str,
        default='action_loss.pkl')
    parser.add_argument(
        '--prop_result_file',
        type=str,
        default="proposals_postNMS.json")
    parser.add_argument(
        '--detect_result_file',
        type=str,
        default="detections_postNMS.json")
    parser.add_argument(
        '--detect_mAP_v_file',
        type=str,
        default="detect_mAP_v.json")
    parser.add_argument(
        '--save_fig_path',
        type=str,
        default="prop_eval_result.jpg")
    parser.add_argument(
        '--detad_sensitivity_file',
        type=str,
        default="detad_sensitivity")

    # Training settings
    parser.add_argument(
        '--train_lr',
        type=float,
        default=0.00005)
    parser.add_argument(
        '--segmentor_lr',
        type=float,
        default=0.0005)
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0001)
    parser.add_argument(
        '--num_epoch',
        type=int,
        default=30)  # org 30
    parser.add_argument(
        '--step_size',
        type=int,
        default=15)
    parser.add_argument(
        '--step_gamma',
        type=float,
        default=0.1)
    parser.add_argument(
        '--match_thres',
        type=float,
        default=0.5)

    # Post processing
    parser.add_argument(
        '--post_process_top_K',
        type=int,
        default=100)
    parser.add_argument(
        '--post_process_thread',
        type=int,
        default=100)
    parser.add_argument(
        '--nms_alpha_detect',
        type=float,
        default=0.85)
    parser.add_argument(
        '--nms_alpha_prop',
        type=float,
        default=0.75)
    parser.add_argument(
        '--nms_lthr_prop',
        type=float,
        default=0.65)
    parser.add_argument(
        '--nms_hthr_prop',
        type=float,
        default=0.9)

    # Model architecture settings
    parser.add_argument(
        '--temporal_scale',
        type=int,
        default=1000)
    parser.add_argument(
        '--prop_temporal_scale',
        type=int,
        default=250)
    parser.add_argument(
        '--boundary_ratio',
        type=float,
        default=0.1)
    parser.add_argument(
        '--feat_dim',
        type=int,
        default=2048)
    parser.add_argument(
        '--bb_hidden_dim',
        type=int,
        default=256)
    parser.add_argument(
        '--ff_out_dim',
        type=int,
        default=256)
    parser.add_argument(
        '--decoder_hidden_dim',
        type=int,
        default=256)
    parser.add_argument(
        '--decoder_out_dim',
        type=int,
        default=256)
    parser.add_argument(
        '--ff_stride',
        type=int,
        default=10)
    parser.add_argument(
        '--decoder_num_classes',
        type=int,
        default=201)


    # Boundary map settings
    parser.add_argument(
        '--num_sample',
        type=int,
        default=32)
    parser.add_argument(
        '--prop_boundary_ratio',
        type=float,
        default=0.5)
    parser.add_argument(
        '--num_sample_perbin',
        type=int,
        default=3)
    parser.add_argument(  # kernel size of 2d convolution on the boundary map
        '--bp_ksize',
        type=int,
        default=3)

    # --------------------------------------------------------------------------------------------------#
    # Performance tuning options
    # --------------------------------------------------------------------------------------------------#
    # Code running options
    parser.add_argument(
        '--model_part',
        default='full',
        type=str,
        choices=['FeatureEnhancer', 'detector', 'full'])

    # method to generate proposals: how to compute the scores; which proposals to use
    parser.add_argument(
        '--post_prop_method',
        default='org_allprop',
        type=str,
        choices=['org_allprop', 'org_selprop', 'org_selpropAct', 'stEd_allprop', 'stEd_selprop', 'action_allprop']) # selproposals: select proposals using start/end scores instead of using all proposals

    parser.add_argument(
        '--RoI_method',
        default='gtad',
        type=str,
        choices=['bmn', 'gtad'])

    parser.add_argument(
        '--binary_actionness',
        default='true',
        type=str,
        choices=['true', 'false'])

    parser.add_argument(   # whether to crop proposals out for the second-stage classification
        '--stage2',
        default='gcn',
        type=str,
        choices=['gcn', 'cnn', 'fc', 'none'])

    parser.add_argument(
        '--gen_prop',
        default='alter_window',
        type=str,
        choices=['all', 'alter_window', 'thr_action', 'unif_rand', 'start_end', 'sliding_window'])

    parser.add_argument(
        '--samp_prop',
        default='rand2',
        type=str,
        choices=['rand3', 'rand2', 'none'])

    parser.add_argument(
        '--iou_thr_method',
        default='fixed',
        type=str,
        choices=['fixed', 'anchor_linear', 'atss'])

    parser.add_argument(
        '--out_prop_map',
        default='false',
        type=str,
        choices=['true', 'false'])

    parser.add_argument(
        '--det_cls_loss',
        default='true',
        type=str,
        choices=['true', 'false'])

    parser.add_argument(
        '--det_reg_loss',
        default='false',
        type=str,
        choices=['true', 'false'])

    parser.add_argument(
        '--eval_det_v',
        default='false',
        type=str,
        choices=['true', 'false'])

    parser.add_argument(
        '--num_samp_prop',
        nargs='+',
        type=int,
        default=[25, 0, 25]) # foregound:middle:background

    # Upper bound for #proposals
    parser.add_argument(
        '--total_num_prop_gp',
        type=int,
        default=7875)  # 7875, 3486, 2016, 1275

    # Control #proposals in alter_win
    parser.add_argument(
        '--step_alter_win_gp',
        type=int,
        default=2)  # 2 (7875), 3 (3486), 4 (2016), 5 (1275)

    # Control #proposals in start_end
    parser.add_argument(
        '--thr_start_end_gp',
        type=float,
        default=0.75)  # 0.8 (8316), 0.81 (7524), 0.85 (4961)

    # Control #proposals in thr_action
    parser.add_argument(
        '--num_thr_action_gp',
        type=int,
        default=20) # 50 (6742), 40 (6652), 30 (6359), 20 (5304)



    parser.add_argument(
        '--loss_ratios',
        nargs='+',
        type=int,
        default=[1, 1, 1])

    parser.add_argument(
        '--nfeat_mode',
        default='feat_ctr',
        type=str,
        choices=['feat_ctr', 'dif_ctr', 'feat'])

    parser.add_argument(
        '--edge_type',
        default='pgcn_iou',
        type=str,
        choices=['grid', 'pgcn_iou', 'pgcn_dist', 'semantic_all', 'semantic_sep'])  # 'semantic_all' and 'semantic_sep' only differ in inference: construct a graph using all proposals or divide them into multiple graphs of the same size of the training graph

    parser.add_argument(
        '--split_gcn',
        default='false',
        type=str,
        choices=['false', 'true'])

    parser.add_argument(
        '--split_temp_edge',
        default='none',
        type=str,
        choices=['in_graph', 'after_graph', 'none'])

    parser.add_argument(
        '--splits',
        type=int,
        default=4)

    parser.add_argument(
        '--num_neigh_split',
        type=int,
        default=4)

    parser.add_argument(
        '--samp_gcn',
        default='sage',
        type=str,
        choices=['none', 'sage']) # used with pgcn_iou, pgcn_dist

    parser.add_argument(
        '--agg_mode',
        default='max',
        type=str,
        choices=['max', 'mean'])

    parser.add_argument(
        '--edge_weight',
        default='true',
        type=str,
        choices=['true', 'false'])

    parser.add_argument(
        '--num_neigh',
        type=int,
        default=4)

    parser.add_argument(
        '--samp_thr',
        nargs='+',
        type=float,
        default=[0.5, 0.6, 0.7])

    parser.add_argument(
        '--infer_score',
        default='cls',
        type=str,
        choices=['reg_cls', 'reg', 'cls'])

    parser.add_argument(
        '--dif_lr',
        default='false',
        type=str,
        choices=['true', 'false'])

    parser.add_argument(
        '--modality',
        default='rgbflow',
        type=str,
        choices=['rgb', 'rgbflow'])


    parser.add_argument(
        '--gcn_aspp',
        default='true',
        type=str,
        choices=['true', 'false'])

    parser.add_argument(
        '--edge_weight_seq',
        default='false',
        type=str,
        choices=['true', 'false'])


    parser.add_argument(
        '--nfeat_mode_seq',
        default='feat_ctr',
        type=str,
        choices=['feat_ctr', 'feat', 'dif_ctr'])

    parser.add_argument(
        '--agg_type_seq',
        default='max',
        type=str,
        choices=['max', 'mean'])

    parser.add_argument(
        '--n_neigh_seq',
        type=int,
        default=4)

    parser.add_argument(
        '--num_levels',
        type=int,
        default=5)

    parser.add_argument(
        '--anchor_scale',
        nargs='+',
        type=int,
        default=[8, 12])

    parser.add_argument(
        '--num_head_layers',
        type=int,
        default=4)

    parser.add_argument(
        '--PBR_actionness',
        default=False,
        action='store_true')

    args = parser.parse_args()

    return args