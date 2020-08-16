# -*- coding: utf-8 -*-
import sys, os
sys.path.append('./Evaluation')
from .eval_proposal import ANETproposal
import multiprocessing as mp
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import json
import numpy as np
from .eval_detection_v import ANETdetectionVideo


def run_evaluation(ground_truth_filename, proposal_filename, 
                   max_avg_nr_proposals=100, 
                   tiou_thresholds=np.linspace(0.5, 0.95, 10),
                   subset='validation'):

    anet_proposal = ANETproposal(ground_truth_filename, proposal_filename,
                                 tiou_thresholds=tiou_thresholds,
                                 max_avg_nr_proposals=max_avg_nr_proposals,
                                 subset=subset, verbose=True, check_status=False)
    anet_proposal.evaluate()
    
    recall = anet_proposal.recall
    average_recall = anet_proposal.avg_recall
    average_nr_proposals = anet_proposal.proposals_per_video
    
    return (average_nr_proposals, average_recall, recall)

def plot_metric(opt, average_nr_proposals, average_recall, recall, tiou_thresholds=np.linspace(0.5, 0.95, 10)):

    fn_size = 14
    plt.figure(num=None, figsize=(12, 8))
    ax = plt.subplot(1,1,1)
    
    colors = ['k', 'r', 'yellow', 'b', 'c', 'm', 'b', 'pink', 'lawngreen', 'indigo']
    area_under_curve = np.zeros_like(tiou_thresholds)
    for i in range(recall.shape[0]):
        area_under_curve[i] = np.trapz(recall[i], average_nr_proposals)

    for idx, tiou in enumerate(tiou_thresholds[::2]):
        ax.plot(average_nr_proposals, recall[2*idx,:], color=colors[idx+1],
                label="tiou=[" + str(tiou) + "], area=" + str(int(area_under_curve[2*idx]*100)/100.), 
                linewidth=4, linestyle='--', marker=None)
    # Plots Average Recall vs Average number of proposals.
    ax.plot(average_nr_proposals, average_recall, color=colors[0],
            label="tiou = 0.5:0.05:0.95," + " area=" + str(int(np.trapz(average_recall, average_nr_proposals)*100)/100.), 
            linewidth=4, linestyle='-', marker=None)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[-1]] + handles[:-1], [labels[-1]] + labels[:-1], loc='best')
    
    plt.ylabel('Average Recall', fontsize=fn_size)
    plt.xlabel('Average Number of Proposals per Video', fontsize=fn_size)
    plt.grid(b=True, which="both")
    plt.ylim([0, 1.0])
    plt.setp(plt.axes().get_xticklabels(), fontsize=fn_size)
    plt.setp(plt.axes().get_yticklabels(), fontsize=fn_size)
    #plt.show()    
    plt.savefig(opt["output"]+'/'+opt["save_fig_path"])

def evaluation_proposal(opt):
    
    uniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid = run_evaluation(
        "./src/Evaluation/data/activity_net_1_3_new.json",
        opt["output"]+'/'+opt["result_file"],
        max_avg_nr_proposals=100,
        tiou_thresholds=np.linspace(0.5, 0.95, 10),
        subset='validation')
    
    plot_metric(opt,uniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid)
    print( "AR@1 is \t",np.mean(uniform_recall_valid[:,0]))
    print( "AR@5 is \t",np.mean(uniform_recall_valid[:,4]))
    print( "AR@10 is \t",np.mean(uniform_recall_valid[:,9]))
    print( "AR@100 is \t",np.mean(uniform_recall_valid[:,-1]))

def _eval_det_video(opt, app, video_id_sublist, result_dict, verbose=False):
    for video_id in video_id_sublist:
        max_ap = app.evaluate_video(video_id)
        result_dict[video_id] = max_ap.mean()


def evaluation_detection_v(opt):
    app = ANETdetectionVideo(ground_truth_filename = opt["video_anno"],
                             prediction_filename = os.path.join(opt["output_path"], opt["detect_result_file"]),
                                 verbose=False, check_status=False)
    video_id_list = app.ground_truth['video-id'].tolist()
    video_id_list = list(set(video_id_list))



    if True:
        global result_dict
        result_dict = mp.Manager().dict()

        num_videos = len(video_id_list)
        num_videos_per_thread = int(num_videos / opt["post_process_thread"])
        processes = []
        # opt["logger"].info("evaluate each video in {} threads".format(opt["post_process_thread"]))

        for tid in range(opt["post_process_thread"] - 1):
            tmp_video_list = video_id_list[tid * num_videos_per_thread:(tid + 1) * num_videos_per_thread]
            p = mp.Process(target=_eval_det_video, args=(opt, app, tmp_video_list, result_dict,))
            p.start()
            processes.append(p)
        tmp_video_list = video_id_list[(opt["post_process_thread"] - 1) * num_videos_per_thread:]
        # print('debug: video_list for last thread is {}'.format(tmp_video_list))
        p = mp.Process(target=_eval_det_video, args=(opt, app, tmp_video_list, result_dict,True))
        p.start()
        processes.append(p)
        for p in processes:
            p.join()

        result_dict = dict(result_dict)

        with open(os.path.join(opt["output_path"], opt["detect_mAP_v_file"]), 'w') as f:
        # with open('output/anet_eval_videos_AmaxAP',"w") as f:
            json.dump(result_dict,f)

    app.verbose=True
    app.evaluate()
