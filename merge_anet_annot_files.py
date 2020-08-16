import json
import pandas as pd
# import h5py
import pickle
#
# video_info = "Data/Activitynet/video_info_new.csv"
# video_anno = 'Data/Activitynet/anet_anno_action.json'
# thumos_anno = "Evaluation/data/thumos_gt.json"
#
#
def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

# df = pd.read_csv(video_info)
# json_data = load_json(video_anno)
#
# for i in range(len(json_data)):
#     video_name = df.video.values[i]
#     video_info = json_data[video_name]
#     video_info['subset'] = df.subset.values[i]
#     video_info['numFrame'] =  int(df.numFrame.values[i])
#     video_info['seconds'] = df.seconds.values[i]
#     video_info['fps'] = df.fps.values[i]
#
#
# with open('anet_gt.json', 'w') as f:
#     json.dump(json_data, f)
#
#
#
#
#
#
# a = "Evaluation/thumos/annot/test_Annotation.csv"
# b = "Evaluation/thumos/annot/thumos14_test_groundtruth.csv"
# c = 'Evaluation/thumos/annot/thumos_gt.json'
#
# aa = pd.read_csv(a)
# bb = pd.read_csv(b)
# cc = load_json(c)
#
# for key, value in cc.items():
#     v_name = key
#     aa[aa.video == key].


# full_path = 'Data/Thumos14/train_data_4000.pkl'
# video_list = pickle.load(open(full_path, 'rb'))
#
# largest_win_end = {} # debug
# new_list = []
# for v_win in video_list:
#     a = 0
#     v_name = v_win['rgb'].split('/')[-1]
#     # debug
#     if v_name not in largest_win_end.keys():
#         largest_win_end[v_name] = [v_win['frames']]
#         new_list.append(v_win)
#     else:
#         for ele in largest_win_end[v_name]:
#             if (ele == v_win['frames']).all():
#                 a = 1
#         if a == 0:
#             largest_win_end[v_name].append(v_win['frames'])
#             new_list.append(v_win)
#
# pickle.dump(new_list, open('Data/Thumos14/new_train_4000.pkl','wb'), pickle.HIGHEST_PROTOCOL)


ground_truth_filename = 'Evaluation/activitynet/annot/activitynet_gt.json'
file = 'Evaluation/activitynet/annot/activity_net_1_3_new.json'
with open(ground_truth_filename, 'r') as fobj:
    data = json.load(fobj)
with open(file, 'r') as fobj:
    ddd = json.load(fobj)

data2 = {}
data2['database'] = data
data2['taxonomy'] = ddd['taxonomy']
data2['version'] =  ddd['version']

outfile = open('activitynet_gt.json', "w")
json.dump(data2, outfile)
outfile.close()

b = 1
