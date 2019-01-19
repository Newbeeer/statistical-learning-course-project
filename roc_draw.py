import numpy as np
from data_generator import mu,datapoint_1,tdatapoint_1,label_train,label_test
import matplotlib.pyplot as plt
from sklearn import metrics


agg_label = np.load('./labels/agg_label.npy')
agg_label = agg_label[:,1]

em_label = np.load('./labels/em_label.npy')
#em_label = em_label[:,1]

fix_majority_label = np.load('./labels/fix_majority_label.npy')
#fix_majority_label = fix_majority_label[:,1]

s_label = np.load('./labels/s_label.npy')
#s_label = s_label[:,1]

m_label = np.load('./labels/m_label.npy')
#m_label = m_label[:,1]


fpr_agg,tpr_agg,thresholds1 =  metrics.roc_curve(label_train,agg_label)
agg=metrics.roc_auc_score(label_train,agg_label)
fpr_em,tpr_em,thresholds2 =  metrics.roc_curve(label_train,em_label)
em = metrics.roc_auc_score(label_train,em_label)
fpr_fix_majority,tpr_fix_majority,thresholds3 =  metrics.roc_curve(label_train,fix_majority_label)
fm = metrics.roc_auc_score(label_train,fix_majority_label)
fpr_s,tpr_s,thresholds4 =  metrics.roc_curve(label_train,s_label)
s = metrics.roc_auc_score(label_train,s_label)
fpr_m,tpr_m,thresholds5 =  metrics.roc_curve(label_train,m_label)
m = metrics.roc_auc_score(label_train,m_label)

plt.plot(fpr_agg,tpr_agg,marker = '.',label="Original EM, AUC="+str(0.951))
plt.plot(fpr_em,tpr_em,marker = '.',label="Adaptive Difficulty EM, AUC="+str(0.975))
plt.plot(fpr_fix_majority,tpr_fix_majority,marker = '.',label="Fixed Difficulty EM, AUC="+str(0.970))
#plt.plot(fpr_s,tpr_s,marker = '.',label="Supervised")
plt.plot(fpr_m,tpr_m,marker = '.',label="Majority Voting, AUC="+str(0.647))
plt.legend(loc='lower right')
plt.title('ROC curve for the estimated true labels')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.savefig('./roc_plot.png')


def train_em():

    # Update train loader after refreshing label
    train_loader_em = torch.utils.data.DataLoader(dataset=train_dataset_em, batch_size=Config.batch_size, shuffle=True)
    data_loader_em = torch.utils.data.DataLoader(dataset=train_dataset_em, batch_size=Config.batch_size, shuffle=False)

    left_model_em.train()

    for batch_idx, (left_data, right_data, label) in enumerate(train_loader_em):

        ep = Variable(right_data).float().cuda()
        images = Variable(left_data).float().cuda()
        label = label.float().cuda()
        left_optimizer_em.zero_grad()
        left_outputs = left_model_em(images)
        right_outputs = right_model_em.forward(ep, label, left_outputs, type=3)
        loss = F.binary_cross_entropy(right_outputs[:, 1].float() + (right_outputs[:, 1] < 0.01).float() * 0.01,label)
        loss.backward()
        left_optimizer_em.step()

    for iter_num in range(2):
        for batch_idx, (left_data, right_data, label) in enumerate(train_loader_em):
            ep = Variable(right_data).float().cuda()
            images = Variable(left_data).float().cuda()
            label = label.float().cuda()

            left_outputs = left_model_em(images)
            right_outputs = right_model_em.forward(ep, label, left_outputs, type=3)

            loss = F.binary_cross_entropy(right_outputs[:, 1].float() + (right_outputs[:, 1] < 0.01).float() * 0.01,label)
            loss.backward()
            right_model_em.weights_update()



    print("E step =====>")
    # E-step
    right_label = []
    for batch_idx, (left_data, right_data, label) in enumerate(data_loader_em):
        ep = Variable(right_data).float().cuda()
        images = Variable(left_data).float().cuda()
        left_outputs = left_model_em(images)
        right_outputs = right_model_em.forward(ep, label, left_outputs, type=3)
        right_outputs = right_outputs[:, 1].float()
        right_label += list(right_outputs.detach().cpu().numpy())
    right_label = torch.FloatTensor(right_label)
    train_dataset_em.label_update(right_label)