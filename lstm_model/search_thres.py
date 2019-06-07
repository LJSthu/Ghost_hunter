import numpy as np
from W_distance import cal_W_distance3

pred_dir = '../data/data4/predict_full_epoch0_iter2100.npy'
label_dir = '../data/data4/label_valid.npy'
predict = np.load(pred_dir)
label = np.load(label_dir)



thres = [0,0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,0.15,0.2]
W = []
for thre in thres:
    distance = cal_W_distance3(predict, label,thre)
    print(thre, distance)



