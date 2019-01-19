import  numpy as np

mu = np.load('mu_1.npy')
print(mu.shape)
for i in range(mu.shape[0]):
    print(mu[i])