import tables
import matplotlib
import numpy as np

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Read hdf5 file
filename = "./final_data/ztraining-4.h5"
h5file = tables.open_file(filename, "r")

training_data = []
eventid = []
channelid = []
labels = []

WaveformTable = h5file.root.Waveform
# GroundTruthTable = h5file.root.GroundTruth

length = WaveformTable.shape[0]
# label_length = GroundTruthTable.shape[0]
print(length)

for i in range(length):
    if i % 10000 == 0:
        print(i)
    Waveform = WaveformTable[i]['Waveform']
    event = WaveformTable[i]['EventID']
    channel = WaveformTable[i]['ChannelID']
    training_data.append(Waveform)
    eventid.append(event)
    channelid.append(channel)
    # if i % 200000 == 0:
    #     training_data2 = np.array(training_data)
    #     train_str = './data/data4/train'+str(i)+'.npy'
    #     np.save(train_str, training_data2)
    #     print(training_data2.shape)

training_data2 = np.array(training_data)
train_str = './final_data/train4/train_full.npy'
np.save(train_str, training_data2)
print(training_data2.shape)

eventid = np.array(eventid)
channelid = np.array(channelid)
assert(eventid.shape[0] == channelid.shape[0])
print(eventid.shape[0])
np.save('./final_data/train4/event.npy', eventid)
np.save('./final_data/train4/channel.npy', channelid)

h5file.close()