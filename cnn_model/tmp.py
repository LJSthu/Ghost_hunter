import numpy as np

data = np.load('./first/predict_full.npy')
print(data.mean(1), data.min(1), data.max(1), data.sum(1))
