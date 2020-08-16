import argparse
import numpy as np
import os

from .eval_detection import ANETdetection

def run_evaluation(ground_truth_filename, prediction_filename,
         subset='validation', tiou_thresholds=np.linspace(0.5, 0.95, 10),
         verbose=True):

    anet_detection = ANETdetection(ground_truth_filename, prediction_filename,
                                   subset=subset, tiou_thresholds=tiou_thresholds,
                                   verbose=verbose, check_status=False)
    anet_detection.evaluate()


def evaluation_detection(opt):

    run_evaluation(ground_truth_filename = opt["eval_anno"],
                   prediction_filename = os.path.join(opt["output_path"], opt["detect_result_file"]),
                   subset='validation', tiou_thresholds=np.linspace(0.5, 0.95, 10))