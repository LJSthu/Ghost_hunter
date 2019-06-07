import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
from model import Model
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
model = Model().cuda()
model.load_state_dict(torch.load('./models/model_epoch0_iter2100.pth'))
train_data = np.load('../data/first/train_full.npy')
train_data = torch.Tensor(train_data)

data = train_data
data = data.unsqueeze(2)
print(data.shape)
data = (data-torch.mean(data, dim=1, keepdim=True))/torch.std(data, dim=1, keepdim=True)
print(data.mean())

length = train_data.shape[0]
print(length)

batch_size = 1024
predict = []
for i in range(length/batch_size + 1):
    if (i+1)*batch_size >= length:
        last = length
    else:
        last = (i+1)*batch_size
    batch_data = data[i*batch_size:last,:,:]
    batch_data = batch_data.cuda()
    pred = model(batch_data)
    pred = pred.cpu().detach().numpy()
    # print(pred.max())
    predict.append(pred)

length = len(predict)
for i in range(length):
    print(i)
    if i == 0:
        final = predict[i]
    else:
        final = np.concatenate((np.array(final), np.array(predict[i])))
predict = np.array(final)
print(predict.shape)
np.save('../data/first/predict_full0.npy',predict)
