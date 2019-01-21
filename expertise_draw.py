import numpy as np
from data_generator import mu,datapoint_1,tdatapoint_1,label_train,label_test
import matplotlib.pyplot as plt
from sklearn import metrics


as_expertise = np.array([[[0.6, 0.4], [0.2, 0.8]], [[0.7, 0.3], [0.4, 0.6]], [[0.6, 0.4], [0.4, 0.6]], [[0.7, 0.3], [0.3, 0.7]], [[0.7, 0.3], [0.4, 0.6]]])

as_expertise_lambda = np.zeros((as_expertise.shape[0],2))

for i in range(as_expertise.shape[0]):
    as_expertise_lambda[i][0] = 4 * np.log(as_expertise[i][0][0] / (1 - as_expertise[i][0][0]))
    as_expertise_lambda[i][1] = 4 * np.log(as_expertise[i][1][1] / (1 - as_expertise[i][1][1]))

m = np.zeros((5,2))
m[:,0] = as_expertise[:,0,0]
m[:,1] = as_expertise[:,1,1]

agg_expertise = np.load('./expertise_agg.npy')
t = np.exp(agg_expertise) / np.expand_dims(np.exp(agg_expertise).sum(2),2)
agg_expertise = np.zeros((5,2))
agg_expertise[:,0] = t[:,0,0]
agg_expertise[:,1] = t[:,1,1]

em_expertise = np.load('./expertise_emdog.npy')
em_expertise = 1/(1+np.exp(-0.25 * em_expertise))

fix_expertise = np.load('./expertise_fixdog.npy')
fix_expertise = 1/(1+np.exp(-0.25 * fix_expertise))

rss_fix = (np.abs(fix_expertise - m)).sum() /5
rss_agg = (np.abs(agg_expertise - m)).sum() /5

print("agg:",rss_agg,"fix:",rss_fix)
plt.scatter(as_expertise[:,0,0],as_expertise[:,1,1],c='b',label = 'True expertise')
txt = ['1','2','3','4','5']

for i in range(as_expertise.shape[0]-1):
    plt.annotate(txt[i], xy = (as_expertise[i,0,0], as_expertise[i,1,1]), xytext = (as_expertise[i,0,0]+0.01, as_expertise[i,1,1]+0.01),arrowprops=dict(facecolor='black',headwidth=0.01, shrink=0.5))
i=4
plt.annotate(txt[i], xy = (as_expertise[i,0,0], as_expertise[i,1,1]), xytext = (as_expertise[i,0,0]-0.01, as_expertise[i,1,1]-0.01),arrowprops=dict(facecolor='black',headwidth=0.01, shrink=0.5))

plt.scatter(agg_expertise[:,0],agg_expertise[:,1],c='r',label = 'Original EM expertise')
for i in range(as_expertise.shape[0]):
    plt.annotate(txt[i], xy = (agg_expertise[i,0], agg_expertise[i,1]), xytext = (agg_expertise[i,0]+0.01, agg_expertise[i,1]+0.01),arrowprops=dict(facecolor='black',headwidth=0.01, shrink=0.5))
'''
plt.scatter(em_expertise[:,0],em_expertise[:,1],c='y',label = 'Adaptive Difficulty EM expertise')
for i in range(as_expertise.shape[0]):
    plt.annotate(txt[i], xy = (em_expertise[i,0], em_expertise[i,1]), xytext = (em_expertise[i,0]+0.01, em_expertise[i,1]+0.01),arrowprops=dict(facecolor='black',headwidth=0.01, shrink=0.5))
'''
plt.scatter(fix_expertise[:,0],fix_expertise[:,1],c='g',label = 'Fixed Difficulty EM expertise')
for i in range(as_expertise.shape[0]):
    plt.annotate(txt[i], xy = (fix_expertise[i,0], fix_expertise[i,1]), xytext = (fix_expertise[i,0]+0.01, fix_expertise[i,1]+0.01),arrowprops=dict(facecolor='black',headwidth=0.01, shrink=0.5))
plt.title('Expertise')
plt.xlabel('sensitivity')
plt.ylabel('specificity')
plt.legend()
plt.savefig('expertise_dog.png')