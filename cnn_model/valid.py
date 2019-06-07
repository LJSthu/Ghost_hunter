from scipy.stats import wasserstein_distance
import math
import torch
import torch.utils.data
import torch.optim as optim
import numpy as np
np.set_printoptions(threshold=np.inf)
from model import Model
from tqdm import *

def cal_W_distance2(predict, truth):
    assert(predict.shape[0] == truth.shape[0])
    length = predict.shape[0]
    distance = 0.0
    for i in tqdm(range(length), desc='calc wdist'):
        pre = []
        weight = []
        if max(predict[i]) <= 0:
            pre = [np.argmax(predict[i].squeeze())]
            weight = 1
        else:
            for j in range(1029):
                if predict[i][j] > 0:
                    pre.append(j)
                    weight.append(predict[i][j])
        pre = np.array(pre)
        weight = np.array(weight)
        label = np.where(truth[i].squeeze() > 0)[0]
        # print(label)
        distance += wasserstein_distance(pre, label, u_weights=weight)
    # print('Wasserstein distance is %f', distance)
    return distance / length

def W_dist(targ, pred):#bs, seq
    print(targ.shape,pred.shape)
    targ = torch.tensor(targ,dtype=torch.double)
    pred = torch.tensor(pred,dtype=torch.double)
    lossC = (targ - pred).abs().sum()
    targ = targ/(torch.sum(targ, dim=1, keepdim=True)+1e-7)
    pred = pred/(torch.sum(pred, dim=1, keepdim=True)+1e-7)
    targA = torch.cumsum(targ, 1)
    predA = torch.cumsum(pred, 1)
    lossB = (targA-predA).abs()
    lossB = lossB.sum(1).mean()
    return lossB

def rebuild(pred,
        lambda_1 = 0.12,
        lambda_2 = 4,
        lambda_3 = .29,
        ):
    return pred**lambda_1*(pred<lambda_3) + pred**lambda_2*(pred>=lambda_3)

if __name__ == '__main__':
    use_cuda = True
    ds_dir = './final_train/'
    model = Model()
    model.load_state_dict(torch.load('models/model_best.pth'))
    if use_cuda:
        model = model.cuda()
    model.eval()

    train_data = np.load(ds_dir + 'train_valid.npy')
    train_label = np.load(ds_dir + 'label_valid.npy')

    train_data = train_data[::200]
    train_label = train_label[::200]

    train_data = torch.Tensor(train_data)

    results = []
    for data in tqdm(train_data.split(100), desc='run nn'):
        if use_cuda:
            data = data.cuda()
        data = data[:, 1:] - data[:, :-1]
        data = data.unsqueeze(1)
        pred = model(data)
        if use_cuda:
            pred = torch.cat([torch.zeros(pred.size(0), 1).cuda(), pred], dim=1)
        else:
            pred = torch.cat([torch.zeros(pred.size(0), 1), pred], dim=1)
        pred = pred.cpu().detach().numpy()
        results.append(pred)
    train_pred_raw = np.concatenate(results, 0)
    print('Original : ', W_dist(train_pred_raw, train_label))
    train_pred = rebuild(train_pred_raw)
    print('Rebuild : ', W_dist(train_pred, train_label))
    lmd = 0.01
    while True:
        train_pred = rebuild(train_pred_raw, lambda_3 = lmd)
        print('Rebuild : %.4f'%lmd, W_dist(train_pred, train_label))
        lmd += 0.01
        if lmd > 1:
            break
