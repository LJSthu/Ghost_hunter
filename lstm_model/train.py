import numpy as np
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
from model import Model, CNNModel
from W_distance import cal_W_distance2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
train_data = np.load('../final_data/train0/train.npy')
train_label = np.load('../final_data/train0/label.npy')
train_data = torch.Tensor(train_data)
train_label = torch.Tensor(train_label)

valid_data = np.load('../final_data/train0/train_valid.npy')
valid_label = np.load('../final_data/train0/label_valid.npy')
valid_data = torch.Tensor(valid_data)
valid_label = torch.Tensor(valid_label)

dataset = torch.utils.data.TensorDataset(
        train_data,
        train_label)
loader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = 512,
        shuffle=True,
        num_workers = 3)

valid_dataset = torch.utils.data.TensorDataset(
    valid_data,
    valid_label)
valid_loader = torch.utils.data.DataLoader(
        dataset = valid_dataset,
        batch_size = 512,
        shuffle = True,
        num_workers = 3
)

model = CNNModel().cuda()
#model.load_state_dict(torch.load('models/model.pth'))
optimizer = optim.Adam(model.parameters(), lr = 1e-5)

best = []
W = []
for epoch in range(10):
    model.train()
    for step, (batch_x, batch_y) in enumerate(loader):
        data = batch_x.unsqueeze(1)
        # data = batch_x.unsqueeze(2)#bs, seq, 1      LSTM MODEL UNCOMMENT THIS
        data = (data-torch.mean(data, dim=2, keepdim=True))/torch.std(data, dim=2, keepdim=True)
        # target = batch_y.unsqueeze(2).cuda()        LSTM MODEL UNCOMMENT THIS
        target = batch_y.cuda()      # LSTML MODEL COMMENT THIS
        data = data.cuda()
        target = target.cuda()
        pred = model(data)
        lossB = torch.abs(pred - target).mean()
        # lossA = -(pred * (target*2-1)).mean()
        lossC = F.cosine_similarity(pred, target)
        loss = torch.exp(-lossC).mean() + lossB
        loss.backward()
        optimizer.step()
        if step % 10 == 0 and step > 0:
            print('%d epoch\'s %d step has total loss %f, the L1 loss is %f'%(epoch, step, loss.item(), lossB.item()))
        if step % 2000 == 0 and step > 0:
            # print( torch.min(pred), torch.max(pred), torch.mean(pred))
            torch.save(model.state_dict(), './models/model_epoch'+str(epoch)+'_iter'+str(step)+'.pth')

            W_distance = 0.0
            model.eval()
            for step, (batch_x, batch_y) in enumerate(valid_loader):

                # data = batch_x.unsqueeze(2).cuda()  # bs, seq, 1  LSTM MODEL UNCOMMENT THIS
                data = batch_x.unsqueeze(1).cuda()
                data = (data - torch.mean(data, dim=2, keepdim=True)) / torch.std(data, dim=2, keepdim=True)
                # target = batch_y.unsqueeze(2).cuda()              LSTM MODEL UNCOMMENT THIS
                target = batch_y.cuda()
                pred = model(data)
                # for d in range(1029):
                #     test_debug = pred.squeeze().cpu().detach()
                #     print(test_debug[10][d])
                #print('debug ', pred.squeeze().cpu().detach().shape, batch_y.squeeze().cpu().detach().shape)
                W_distance += cal_W_distance2(pred.squeeze().cpu().detach(), batch_y.squeeze().cpu().detach())
            if (W_distance/len(valid_loader) < 10):
                best.append(epoch*10000+step)

            W.append(W_distance/len(valid_loader))
            print('epoch %d W_distance is %f\n' % (epoch, W_distance/len(valid_loader)))
            model.train()

print(W)
print(best)