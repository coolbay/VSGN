
from Evaluation.activitynet.prop_post_processing import post_processing_multiproc as post_processing
from Evaluation.activitynet.get_prop_performance import evaluation_proposal
from Evaluation.activitynet.generate_detection import gen_detections_multiproc as gen_det_anet
from Evaluation.activitynet.get_detect_performance import evaluation_detection as eval_det_anet
from Evaluation.activitynet.get_detect_performance_v import evaluation_detection_v
from DETAD.sensitivity_analysis import detad_analyze_anet

from Evaluation.thumos.generate_detection import gen_detection_multicore as gen_det_thumos
from Evaluation.thumos.get_detect_performance import evaluation_detection as eval_det_thumos

from Evaluation.hacs.generate_detection import gen_detections_multiproc as gen_det_hacs
from Evaluation.hacs.get_detect_performance import evaluation_detection as eval_det_hacs

import Utils.opts as opts
import os
import datetime
import subprocess

if __name__ == '__main__':

    opt = opts.parse_opt()
    opt = vars(opt)

    print(opt)

    if not os.path.exists(opt["output_path"]):
        print('No predictions! Please run inference first!')


    if opt['dataset'] == 'activitynet':
        # 2. Run NMS and test the average recall of the obtained proposals
        print(datetime.datetime.now())
        print("---------------------------------------------------------------------------------------------")
        print("2. Proposal evaluation starts!")
        print("---------------------------------------------------------------------------------------------")

        print("a. Post-process proposals!")
        post_processing(opt)

        print("b. Evaluate proposals!")
        evaluation_proposal(opt)

        print("Proposal evaluation finishes! \n")

        # 3. Combine with the video-level classification scores and test the mAP of the detections

        print("---------------------------------------------------------------------------------------------")
        print("3. Detection evaluation starts!")
        print("---------------------------------------------------------------------------------------------")

        print("a. Generate detections!")
        gen_det_anet(opt)

        print("b. Evaluate the detection results!")
        eval_det_anet(opt)
        print("Detection evaluation finishes! \n")

        if opt['eval_det_v'] == 'true':
            print("c. Evaluate detection for each video!")
            evaluation_detection_v(opt)
            print("Detection evaluation finishes! \n")

        # 4. Run DETAD to test sensitivity to lengths, duration and instances
        print(datetime.datetime.now())
        print("---------------------------------------------------------------------------------------------")
        print("4. DETAD evaluation starts!")
        print("---------------------------------------------------------------------------------------------")
        detad_analyze_anet(opt)
        print("DETAD evaluation finishes! \n")

    elif opt['dataset'] == 'thumos':

        print("---------------------------------------------------------------------------------------------")
        print("3. Detection evaluation starts!")
        print("---------------------------------------------------------------------------------------------")

        print("a. Generate detections!")
        gen_det_thumos(opt)

        print("b. Evaluate the detection results!")
        eval_det_thumos(opt)
        print("Detection evaluation finishes! \n")

    elif opt['dataset'] == 'hacs':


        print(datetime.datetime.now())
        print("---------------------------------------------------------------------------------------------")
        print("3. Detection evaluation starts!")
        print("---------------------------------------------------------------------------------------------")

        print("a. Generate detections!")
        gen_det_hacs(opt)

        print("b. Evaluate the detection results!")
        eval_det_hacs(opt)
        print("Detection evaluation finishes! \n")

        if opt['eval_det_v'] == 'true':
            print("c. Evaluate detection for each video!")
            evaluation_detection_v(opt)
            print("Detection evaluation finishes! \n")


