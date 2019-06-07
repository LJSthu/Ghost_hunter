#-*-coding:utf-8-*-
import tables
import numpy as np


# Read hdf5 file
filename = "./final_data/ztraining-4.h5"
h5file = tables.open_file(filename, "r")

training_data = []
labels = []

GroundTruthTable = h5file.root.GroundTruth
label_length = GroundTruthTable.shape[0]
print(label_length)

# label_length = 35000

eventid = np.load('./final_data/train4/event.npy')
channelid = np.load('./final_data/train4/channel.npy')


number_of_label = -1
index = 0
judge = True
pre_event = -1
pre_channel = -1
label = np.zeros(1029)
for i in range(label_length):
    if i % 100000 == 0:
        print(i)
    x = GroundTruthTable[i]
    # print(x)
    event = x['EventID']
    channel = x['ChannelID']
    if channel == pre_channel and event == pre_event:   # 还是之前图的标签
        index = int(x['PETime'])
        if index > 1028:
            index = 1028
        label[index] += 1
    else:                      # 新的波形图的标签
        number_of_label += 1
        labels.append(label)

        if (not event == eventid[number_of_label]) or (not channel == channelid[number_of_label]):
            print('error occurs ', eventid[number_of_label], channelid[number_of_label])
            label = np.zeros(1029)
            labels.append(label)
            number_of_label += 1

        label = np.zeros(1029)
        pre_event = event
        pre_channel = channel
        index = int(x['PETime'])
        if index > 1028:
            index = 1028
        label[index] += 1


    # if i % 5000000 == 0:
    #     labels2 = np.array(labels)
    #     label_str = './final_data/train2/label' + str(i) + '.npy'
    #     np.save(label_str, labels2)
    #     print(labels2.shape)

labels.append(label)
labels2 = np.array(labels)
labels2 = labels2[1:,:]
label_str = './final_data/train4/label_full.npy'
np.save(label_str, labels2)
print(labels2.shape)

h5file.close()