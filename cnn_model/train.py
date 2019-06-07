import numpy as np
import random
import torch
import torch.utils.data
import torch.optim as optim
from model import Model
import sys
np.set_printoptions(threshold=sys.maxsize)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

use_cuda = True
mini_batch = False

ds_dir = './final_train/'

train_data = np.load(ds_dir + 'train_train.npy')
train_label = np.load(ds_dir + 'label_train.npy')
_valid_data = np.load(ds_dir + 'train_valid.npy')
_valid_label = np.load(ds_dir + 'label_valid.npy')
print('valid_shape = %s'%str(_valid_data.shape))

def rebuild_valid():
    global valid_data
    global valid_label
    global valid_data_2
    global valid_label_2
    global valid_input
    global valid_input_2
    global best_loss
    best_loss = 1e10
    base = random.randint(0,len(_valid_data)-1-20000)
    print('rebuild valid! <%d>'%base)
    valid_start_index = base + 5000
    valid_split_index = base + 8000
    valid_end_index = base + 10500
    valid_data = torch.Tensor(_valid_data)[valid_start_index:valid_split_index]
    valid_label = torch.Tensor(_valid_label)[valid_start_index:valid_split_index]
    valid_data_2 = torch.Tensor(_valid_data)[valid_split_index:valid_end_index]
    valid_label_2 = torch.Tensor(_valid_label)[valid_split_index:valid_end_index]
    valid_input = (valid_data[:, 1:] - valid_data[:, :-1]).unsqueeze(1)
    valid_input_2 = (valid_data_2[:, 1:] - valid_data_2[:, :-1]).unsqueeze(1)
rebuild_valid()

train_data = torch.Tensor(train_data)
train_label = torch.Tensor(train_label)

dataset = torch.utils.data.TensorDataset(
        train_data,
        train_label)

loader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = 1,
        shuffle=True,
        num_workers = 3,
        drop_last = True
        )

model = Model()
if use_cuda:
    model = model.cuda()
#model.train()
model.eval()
model._initialize_weights()
try:
    model.load_state_dict(torch.load('models/model_best.pth'))
except RuntimeError:
    print('problem loading models!')
    pass
optimizer = optim.Adam(model.parameters(), lr = 1e-7, betas=(0.9, 0.9))

def WLoss(targ, pred):#bs, seq
    #lossC = (targ - pred).abs().sum()
    targ = targ/(torch.sum(targ, dim=1, keepdim=True)+1e-7)
    pred = pred/(torch.sum(pred, dim=1, keepdim=True)+1e-7)
    targA = torch.cumsum(targ, 1)
    predA = torch.cumsum(pred, 1)
    lossB = (targA-predA).abs()
    lossB = lossB.sum(1).mean()
    return lossB, lossB
    return lossB*100 + lossC, lossB

criterion = torch.nn.L1Loss()
best_loss = 1e10
need_to_restart = 30
need_to_rebase = 0
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for epoch in range(30):
    for step, (batch_x, batch_y) in enumerate(loader):
        if need_to_restart > 20:
            try:
                model.load_state_dict(torch.load('models/model_best.pth'))
            except RuntimeError:
                print('error load model!')
            need_to_restart = 0
            need_to_rebase += 1
            print('\033[31mRestart!\033[0m')
        if need_to_rebase > 5:
            print('\033[31mRebase!\033[0m')
            rebuild_valid()
            need_to_rebase = 0
        if mini_batch:
            _batch_x = []
            _batch_y = []
            for _bx,_by in zip(batch_x, batch_y):
                while True:
                    pos = random.randint(0,1029-100)
                    if _by[pos:pos+50].sum():
                        break
                _batch_x.append(_bx[pos:pos+50]-_bx[pos+1:pos+51])
                _batch_y.append(_by[pos:pos+50])
            batch_x = torch.stack(_batch_x)
            batch_y = torch.stack(_batch_y)
        if use_cuda:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        batch_x = batch_x[:,1:] - batch_x[:,:-1]
        batch_y = batch_y[:,1:]
        data = batch_x.unsqueeze(1)#bs, 1, seq
        target = batch_y
        pred, bloss = model(data)
        wloss, _ = WLoss(target, pred)
        loss = wloss + bloss
        loss.backward()
        optimizer.step()
        if step % 2000 == 0:
            print('INFO : ',epoch, step, loss)
            print('bloss = ',bloss, 'wloss = ', wloss)
        if step % 500 == 499:
            torch.save(model.state_dict(), 'models/model.pth')
        if step % 4000 == 0:
            tpred = []
            for data in valid_input.split(50):
                if use_cuda:
                    data = data.cuda()
                _tpred, _bloss = model(data)
                _tpred = _tpred.cpu().detach()
                _tpred = torch.cat([torch.zeros(_tpred.size(0), 1), _tpred], dim=1)
                tpred.append(_tpred)
            tpred = torch.cat(tpred, 0)
            _, tloss = WLoss(valid_label, tpred)
            if tloss < best_loss:
                best_loss = tloss
                torch.save(model.state_dict(), 'models/model_best.pth')
                print("\033[32mBest model!"+ str(tloss)+'\033[0m')
                need_to_restart = 0
            else:
                print("\033[33mNot the best."+str(tloss)+'\033[0m')
                need_to_restart += 1
        if step % 8000 == 0:
            model.eval()
            bloss = []
            tpred = []
            for data in valid_input_2.split(50):
                if use_cuda:
                    data = data.cuda()
                _tpred, _bloss = model(data)
                _tpred = _tpred.cpu().detach()
                _bloss = _bloss.cpu().detach()
                _tpred = torch.cat([torch.zeros(_tpred.size(0), 1), _tpred], dim=1)
                tpred.append(_tpred)
                bloss.append(_bloss)
            tpred = torch.cat(tpred, 0)
            bloss = torch.stack(bloss, 0).mean()
            _, tloss = WLoss(valid_label_2, tpred)
            print("\033[34mLevel2 valid : "+ str(tloss)+' , bloss = '+str(bloss)+'\033[0m')
            #model.train()
