import numpy as np

np.random.seed(1228)
datapoint_1 = np.zeros((1000,3))
tdatapoint_1 = np.zeros((1000,3))
label_train = np.ones(1000)
mu = np.ones(1000)
label_test = np.ones(1000)
label_train[500:1000] -= 1
label_test[500:1000] -= 1

w = np.array((1,1,0)).reshape((3,1))
mean = 1
#datapoint_1 ~ N((2,2)^T,I)
datapoint_1[:500, 0] = np.random.randn(500) * 1 + mean
datapoint_1[:500, 1] = np.random.randn(500) * 1 + mean
datapoint_1[:500, 2] = 1

datapoint_1[500:1000, 0] = np.random.randn(500) * 1 - mean
datapoint_1[500:1000, 1] = np.random.randn(500) * 1 - mean
datapoint_1[500:1000, 2] = 1

tdatapoint_1[:500, 0] = np.random.randn(500) * 1 + mean
tdatapoint_1[:500, 1] = np.random.randn(500) * 1 + mean
tdatapoint_1[:500, 2] = 1

tdatapoint_1[500:1000, 0] = np.random.randn(500) * 1 - mean
tdatapoint_1[500:1000, 1] = np.random.randn(500) * 1 - mean
tdatapoint_1[500:1000, 2] = 1

mu[:500] = (1 / (1 + np.exp(-1 * datapoint_1[:500].dot(w)))).reshape(500)
mu[500:1000] = (1 / (1 + np.exp(-1 * datapoint_1[500:1000].dot(w)))).reshape(500)

