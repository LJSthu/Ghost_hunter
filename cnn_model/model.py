import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetBlock(torch.nn.Module):
    def __init__(self,
            hidden_size = 20,
            kernel_size = 16,
            use_dropout = True,
            norm_layer=nn.InstanceNorm1d, 
            ):
        super(ResnetBlock, self).__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.use_dropout = use_dropout
        self.convA = torch.nn.ConvTranspose1d(self.hidden_size, self.hidden_size, 16)
        self.convB = torch.nn.Conv1d(self.hidden_size, self.hidden_size,16)
        if use_dropout:
            self.dropout = torch.nn.Dropout(0.5)
        if not norm_layer is None:
            self.norm = norm_layer(self.hidden_size)

    def forward(self, data):
        _data = F.relu(self.convA(data))
        data = data + self.norm(self.convB(_data))
        return data

class Model(torch.nn.Module):
    def __init__(self,
            hidden_size = 32
            ):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.layer0 = torch.nn.ConvTranspose1d(1, self.hidden_size, 8)
        self.layer1 = ResnetBlock(self.hidden_size, 16)
        self.layer2 = ResnetBlock(self.hidden_size, 16)
        self.layer3 = ResnetBlock(self.hidden_size, 8)
        self.layer4 = torch.nn.Conv1d(self.hidden_size, 1, 8)
        self.layer4_ = torch.nn.ConvTranspose1d(1, self.hidden_size, 8)
        self.layer3_ = ResnetBlock(self.hidden_size, 16)
        self.layer2_ = ResnetBlock(self.hidden_size, 16)
        self.layer1_ = ResnetBlock(self.hidden_size, 8)
        self.layer0_ = torch.nn.Conv1d(self.hidden_size, 1, 8)

    def forward(self, data):
        _data = data
        data = F.tanh(self.layer0(data))
        data = self.layer1(data)
        data = self.layer2(data)
        data = self.layer3(data)
        data = F.sigmoid(self.layer4(data))
        pred = data.squeeze(1)
        data = F.tanh(self.layer4_(data))
        data = self.layer3_(data)
        data = self.layer2_(data)
        data = self.layer1_(data)
        data = self.layer0_(data)
        loss = (data - _data).abs().mean()
        return pred, loss

    def _initialize_weights(self):
	for m in self.modules():
	    if isinstance(m, nn.Conv2d):
		n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
		m.weight.data.normal_(0, math.sqrt(2. / n))
		if m.bias is not None:
		    m.bias.data.zero_()
	    elif isinstance(m, nn.BatchNorm2d):
		m.weight.data.fill_(1)
		m.bias.data.zero_()
	    elif isinstance(m, nn.Linear):
		m.weight.data.normal_(0, 0.01)
		m.bias.data.zero_()
