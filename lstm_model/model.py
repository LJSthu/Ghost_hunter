import torch
import torch.nn.functional as F
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm = torch.nn.LSTM(1, 10, 1, batch_first = True, bidirectional = True)
        self.final = torch.nn.Linear(20, 1)
        self.weight_init()
    def weight_init(self):
        nn.init.kaiming_uniform_(self.final.weight, mode='fan_in')
    def forward(self, data):
        data = self.lstm(data)[0]
        data = self.final(data)
        # data = F.leaky_relu(data, negative_slope=0.1)
        return data



class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.hidden_size = 20
        self.convA1 = torch.nn.ConvTranspose1d(1, self.hidden_size, 16)
        self.convB1 = torch.nn.Conv1d(self.hidden_size, self.hidden_size,16)
        self.convA2 = torch.nn.ConvTranspose1d(self.hidden_size, self.hidden_size, 16)
        self.convB2 = torch.nn.Conv1d(self.hidden_size, self.hidden_size,16)
        self.convA3 = torch.nn.ConvTranspose1d(self.hidden_size, self.hidden_size, 16)
        self.convB3 = torch.nn.Conv1d(self.hidden_size, 1,16)
        self.linear = torch.nn.Linear(1029, 1029)
        self.weight_init()

    def forward(self, data):
        data = F.tanh(self.convA1(data))
        data = F.tanh(self.convB1(data))
        data = self.dropout1(data)
        data = F.tanh(self.convA2(data))
        data = F.tanh(self.convB2(data))
        data = self.dropout2(data)
        data = F.tanh(self.convA3(data))
        data = F.tanh(self.convB3(data))
        #print('data2: ', data.abs().max())
        data = data.squeeze(1)
        return data

    def weight_init(self):
        nn.init.xavier_normal_(self.convA1.weight)
        nn.init.xavier_normal_(self.convA2.weight)
        nn.init.xavier_normal_(self.convA3.weight)
        nn.init.xavier_normal_(self.convB1.weight)
        nn.init.xavier_normal_(self.convB2.weight)
        nn.init.xavier_normal_(self.convB3.weight)
        nn.init.xavier_normal_(self.linear.weight)

