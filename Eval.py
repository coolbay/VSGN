
from Evaluation.thumos.generate_detection import gen_detection_multicore as gen_det_thumos
from Evaluation.thumos.get_detect_performance import evaluation_detection as eval_det_thumos
import Utils.opts as opts
import os

if __name__ == '__main__':

    opt = opts.parse_opt()
    opt = vars(opt)

    print(opt)

    if not os.path.exists(opt["output_path"]):
        print('No predictions! Please run inference first!')

    print("---------------------------------------------------------------------------------------------")
    print("3. Detection evaluation starts!")
    print("---------------------------------------------------------------------------------------------")

    print("a. Generate detections!")
    gen_det_thumos(opt)

    print("b. Evaluate the detection results!")
    eval_det_thumos(opt)
    print("Detection evaluation finishes! \n")
