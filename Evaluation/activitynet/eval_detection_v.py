

import numpy as np
from .eval_detection import ANETdetection, compute_average_precision_detection

class ANETdetectionVideo(ANETdetection):

    GROUND_TRUTH_FIELDS = ['database', 'taxonomy', 'version']
    PREDICTION_FIELDS = ['results', 'version', 'external_data']

    def __init__(self, ground_truth_filename=None, prediction_filename=None,
                 ground_truth_fields=GROUND_TRUTH_FIELDS,
                 prediction_fields=PREDICTION_FIELDS,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                 subset='validation', verbose=False,
                 check_status=True):
        super(ANETdetectionVideo, self).__init__(ground_truth_filename, prediction_filename,
                 ground_truth_fields,
                 prediction_fields,
                 tiou_thresholds,
                 subset, verbose,
                 check_status)

    def wrapper_compute_map_video(self, video_id):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index.items())))
        for activity, cidx in self.activity_index.items():
            gt_idx = (self.ground_truth['label'] == cidx)  &  (self.ground_truth['video-id'] == video_id)
            pred_idx = (self.prediction['label'] == cidx)  &  (self.prediction['video-id'] == video_id)
            ap[:,cidx] = compute_average_precision_detection(
                self.ground_truth.loc[gt_idx].reset_index(drop=True),
                self.prediction.loc[pred_idx].reset_index(drop=True),
                tiou_thresholds=self.tiou_thresholds)
        return ap.max(axis=1)

    def evaluate_video(self, video_id="eUKMPNZ3NI4", verbose=True):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        # self.ap = self.wrapper_compute_average_precision()
        self.mAP = self.wrapper_compute_map_video(video_id)
        if self.verbose and verbose:
            print('[RESULTS]\tAverage-maxAP: {} for video {}'.format(self.mAP.mean(),video_id))

        return self.mAP
