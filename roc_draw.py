import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

label_train = np.load('./labels/correct_label.npy')
print(label_train.shape)
agg_label = np.load('./labels/agg_label.npy')

agg_label = agg_label

em_label = np.load('./labels/em_label.npy')
#em_label = em_label[:,1]

fix_majority_label = np.load('./labels/fix_majority_label.npy')
print(fix_majority_label.shape)
#fix_majority_label = fix_majority_label[:,1]

s_label = np.load('./labels/s_label.npy')
#s_label = s_label[:,1]

m_label = np.load('./labels/m_label.npy')
print(m_label,m_label.shape)
#m_label = m_label[:,1]


fpr_agg,tpr_agg,thresholds1 =  metrics.roc_curve(label_train,agg_label)
agg=metrics.roc_auc_score(label_train,agg_label)
fpr_em,tpr_em,thresholds2 =  metrics.roc_curve(label_train,em_label)
em = metrics.roc_auc_score(label_train,em_label)
fpr_fix_majority,tpr_fix_majority,thresholds3 =  metrics.roc_curve(label_train,fix_majority_label)
fm = metrics.roc_auc_score(label_train,fix_majority_label)
print(label_train.shape,m_label.shape)
fpr_m,tpr_m,thresholds5 =  metrics.roc_curve(label_train,m_label)

m = metrics.roc_auc_score(label_train,m_label)

plt.plot(fpr_agg,tpr_agg,marker = '.',label="Original EM, AUC="+str(0.796))
plt.plot(fpr_em,tpr_em,marker = '.',label="Adaptive Difficulty EM, AUC="+str(0.796))
plt.plot(fpr_fix_majority,tpr_fix_majority,marker = '.',label="Fixed Difficulty EM, AUC="+str(0.853))
#plt.plot(fpr_s,tpr_s,marker = '.',label="Supervised")
plt.plot(fpr_m,tpr_m,marker = '.',label="Majority Voting, AUC="+str(0.500))
plt.legend(loc='lower right')
plt.title('ROC curve for the estimated true labels')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.savefig('./roc_plot_dog.png')

