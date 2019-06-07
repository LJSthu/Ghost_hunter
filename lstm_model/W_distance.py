from scipy.stats import wasserstein_distance
import numpy as np


dir = './data/data4/'


def cal_W_distance1(dir):
    predict = np.load(dir + 'predict.npy')
    truth = np.load(dir + 'label.npy')
    length = predict.shape[0]
    distance = 0.0
    for i in range(length):
        distance += wasserstein_distance(predict[i], truth[i])
    # print('Wasserstein distance is %f', distance)
    return distance


def cal_W_distance2(predict, truth):
    # print(predict.shape)
    assert(predict.shape[0] == truth.shape[0])
    length = predict.shape[0]
    # predict = np.maximum(predict, 1e-5)

    distance = 0.0
    for i in range(length):
        pre = []
        weight = []
        if max(predict[i]) <= 0:
            pre = [np.argmax(predict[i].squeeze())]
            weight.append(1)
        else:
            for j in range(1029):
                if predict[i][j] > 0:
                    pre.append(j)
                    weight.append(predict[i][j])
        pre = np.array(pre)
        if len(weight) > 0:
            weight = np.array(weight)
            label = np.where(truth[i].squeeze() > 0)[0]
            if weight.shape[0] > 0:
                distance += wasserstein_distance(pre, label, u_weights=weight)
    # print('Wasserstein distance is %f', distance)
    return distance / length

def cal_W_distance3(predict, truth, thres):
    assert(predict.shape[0] == truth.shape[0])
    length = predict.shape[0]
    # predict = np.maximum(predict, 1e-5)

    distance = 0.0
    for i in range(length):
        pre = []
        weight = []
        if max(predict[i]) <= thres:
            pre = [np.argmax(predict[i].squeeze())]
            weight = 1
        else:
            for j in range(1029):
                if predict[i][j] > thres:
                    pre.append(j)
                    weight.append(predict[i][j])
        pre = np.array(pre)
        weight = np.array(weight)
        label = np.where(truth[i].squeeze() > 0)[0]
        # print(label)
        distance += wasserstein_distance(pre, label, u_weights=weight)
    # print('Wasserstein distance is %f', distance)
    return distance / length
