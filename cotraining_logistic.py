"""
cotraining - main file for training and testing
"""
import torch
import torch.nn.functional as F
from model import right_model_em,right_model_agg,right_model_fix
from model_logistic import left_model_em,left_model_agg,left_model_fix,left_model_supervised,left_model_majority
from data_logistic import *
from util import *
from torch.autograd import Variable
from  common import  args,Config
from tqdm import tqdm
import numpy as np
from sklearn.metrics import  roc_auc_score

torch.cuda.set_device(Config.device_id)


def train_reparameterization():

    # Update train loader after refreshing label
    train_loader_em = torch.utils.data.DataLoader(dataset=train_dataset_em, batch_size=1000, shuffle=False)

    left_model_em.train()
    mu_all = []
    left_outputs_all = []
    data = []
    for batch_idx, (left_data, right_data, label) in enumerate(train_loader_em):
        images = Variable(left_data).float().cuda()
        label = label.float().cuda()
        left_outputs = left_model_em(images)
        data += list(left_data.detach().cpu().numpy())
        left_outputs_all += list(left_outputs[:, 1].detach().cpu().numpy())
        mu_all += list(label.detach().cpu().numpy())
        sum = torch.sum(label[:500] > 0.5)+torch.sum(label[500:1000] < 0.5)
        print(" EM:current training acc:%f " %(float((sum*1.0))/1000.0))


    mu_all = torch.FloatTensor(mu_all)
    np.save('./labels/em_label.npy', mu_all.detach().cpu().numpy())
    data = torch.FloatTensor(data)
    left_outputs_all = torch.FloatTensor(left_outputs_all)
    left_model_em.update_weights(mu_all,data,left_outputs_all)

    for iter_num in range(1000):
        for batch_idx, (left_data, right_data, label) in enumerate(train_loader_em):
            ep = Variable(right_data).float().cuda()
            images = Variable(left_data).float().cuda()
            label = label.float().cuda()

            left_outputs = left_model_em(images)
            right_outputs = right_model_em.forward(ep, label, left_outputs, type=3)

            loss = F.binary_cross_entropy(right_outputs[:, 1].float() + (right_outputs[:, 1] < 0.01).float() * 0.01, label)
            loss.backward()
            right_model_em.weights_update()

    # E-step
    right_label = []
    for batch_idx, (left_data, right_data, label) in enumerate(train_loader_em):
        ep = Variable(right_data).float().cuda()
        images = Variable(left_data).float().cuda()
        left_outputs = left_model_em(images)
        right_outputs = right_model_em.forward(ep, label, left_outputs, type=3)
        right_outputs = right_outputs[:, 1].float()
        right_label += list(right_outputs.detach().cpu().numpy())
    right_label = torch.FloatTensor(right_label)
    train_dataset_em.label_update(right_label)


def train_agg(epoch,mu=0):
    # Learning from crowds

    # training
    left_model_agg.train()
    right_model_agg.train()

    right_outputs_all = []
    ep_label_all = []
    left_outputs_all = []
    data = []

    for batch_idx, (left_data, right_data, label) in enumerate(data_loader_agg):

        ep = Variable(right_data).float().cuda()
        images = Variable(left_data).float().cuda()
        left_outputs = left_model_agg(images)
        data += list(left_data.detach().cpu().numpy())
        left_outputs_all += list(left_outputs[:, 1].detach().cpu().numpy())
        right_outputs = right_model_agg(ep, 1, left_outputs, type=3)

        sum = torch.sum(right_outputs[:500,1] > 0.5) + torch.sum(right_outputs[500:1000,1] < 0.5)
        # E step
        right_outputs = list(right_outputs.detach().cpu().numpy())
        right_outputs_all += right_outputs

        expert_label = torch.zeros((ep.size()[0], ep.size()[1])).cuda()
        max_element, expert_label[:, :] = torch.max(ep, 2)
        mask = (max_element == 0)
        expert_label = expert_label + (-Config.num_classes) * mask.float().cuda()
        expert_label = list(expert_label.detach().cpu().numpy())
        ep_label_all += expert_label

        print("Agg:current training acc:%f " % (float((sum * 1.0)) / 1000.0))


    if epoch == 0:
        right_outputs_all = torch.stack([mu,mu],1)

    mu_all = torch.FloatTensor(right_outputs_all)
    np.save('./labels/agg_label.npy', mu_all.detach().cpu().numpy())

    data = torch.FloatTensor(data)
    left_outputs_all = torch.FloatTensor(left_outputs_all)
    ep_label_all = torch.FloatTensor(ep_label_all)
    # M-step
    left_model_agg.update_weights(mu_all[:,1], data, left_outputs_all)
    expert_parameters = M_step(ep_label_all, mu_all)
    right_model_agg.weights_update(expert_parameters)

def train_supervised():

    # training
    left_model_supervised.train()
    left_model_majority.train()
    right_outputs_s = []
    right_outputs_m = []
    left_outputs_s = []
    left_outputs_m = []
    data = []
    for batch_idx, (left_data, right_data, label) in enumerate(data_loader_supervised):

        images = Variable(left_data).float().cuda()
        left_outputs = left_model_supervised(images)
        data += list(left_data.detach().cpu().numpy())
        left_outputs_s += list(left_outputs[:, 1].detach().cpu().numpy())
        sum_s = torch.sum(left_outputs[:500,1] > 0.5) + torch.sum(left_outputs[500:1000,1] < 0.5)


        left_outputs = left_model_majority(images)
        left_outputs_m += list(left_outputs[:, 1].detach().cpu().numpy())
        sum_m = torch.sum(left_outputs[:500,1] > 0.5) + torch.sum(left_outputs[500:1000,1] < 0.5)

        right_outputs_s += list(label.detach().cpu().numpy())
        right_outputs_m += list(torch.max(torch.sum(right_data, 1),1)[1].detach().cpu().numpy())


        print("Supervised: current training acc:%f " % (float((sum_s * 1.0)) / 1000.0))
        print("Majority: current training acc:%f " % (float((sum_m * 1.0)) / 1000.0))

    # Update
    mu_s = torch.FloatTensor(right_outputs_s)
    left_outputs_s = torch.FloatTensor(left_outputs_s)
    left_outputs_m = torch.FloatTensor(left_outputs_m)
    data = torch.FloatTensor(data)
    np.save('./labels/s_label.npy', mu_s.detach().cpu().numpy())
    left_model_supervised.update_weights(mu_s, data, left_outputs_s)
    mu_m = torch.FloatTensor(right_outputs_m)

    np.save('./labels/m_label.npy', mu_m.detach().cpu().numpy())
    left_model_majority.update_weights(mu_m, data, left_outputs_m)


def train_fix_major():

    # Update train loader after refreshing label
    train_loader_fix = torch.utils.data.DataLoader(dataset=train_dataset_fix, batch_size=1000, shuffle=False)

    left_model_fix.train()
    mu_all = []
    left_outputs_all = []
    data = []
    for batch_idx, (left_data, right_data, label, fix_label) in enumerate(train_loader_fix):
        images = Variable(left_data).float().cuda()
        label = label.float().cuda()
        left_outputs = left_model_fix(images)
        data += list(left_data.detach().cpu().numpy())
        left_outputs_all += list(left_outputs[:, 1].detach().cpu().numpy())
        mu_all += list(label.detach().cpu().numpy())
        sum = torch.sum(label[:500] > 0.5)+torch.sum(label[500:1000] < 0.5)
        print("Fix major:current training acc:%f " %(float((sum*1.0))/1000.0))


    mu_all = torch.FloatTensor(mu_all)
    np.save('./labels/fix_majority_label.npy', mu_all.detach().cpu().numpy())
    data = torch.FloatTensor(data)
    left_outputs_all = torch.FloatTensor(left_outputs_all)
    left_model_fix.update_weights(mu_all,data,left_outputs_all)

    for iter_num in range(1000):
        for batch_idx, (left_data, right_data, label, fix_label) in enumerate(train_loader_fix):
            ep = Variable(right_data).float().cuda()
            images = Variable(left_data).float().cuda()
            label = label.float().cuda()

            left_outputs = left_model_fix(images)
            right_outputs = right_model_fix.forward(ep, fix_label[:, 1], left_outputs, type=3)
            loss = F.binary_cross_entropy(right_outputs[:, 1].float() + (right_outputs[:, 1] < 0.01).float() * 0.01, label)
            loss.backward()
            right_model_fix.weights_update()

    # E-step
    right_label = []
    for batch_idx, (left_data, right_data, label, fix_label) in enumerate(train_loader_fix):
        ep = Variable(right_data).float().cuda()
        images = Variable(left_data).float().cuda()
        left_outputs = left_model_fix(images)
        right_outputs = right_model_fix.forward(ep, fix_label[:, 1], left_outputs, type=3)
        right_outputs = right_outputs[:, 1].float()
        right_label += list(right_outputs.detach().cpu().numpy())
    right_label = torch.FloatTensor(right_label)
    train_dataset_fix.label_update(right_label)

def Initial_E_em():

    right_label = []
    data_loader = torch.utils.data.DataLoader(dataset=train_dataset_em, batch_size=1000, shuffle=False)
    print("Initial E step =====>")
    for batch_idx, (left_data, right_data, label) in enumerate(tqdm(data_loader)):
        ep = Variable(right_data).float().cuda()
        right_outputs = right_model_em.forward(ep, label, ep, type=0)
        right_outputs = right_outputs[:, 1].float()
        right_label += list(right_outputs.detach().cpu().numpy())

    right_label = torch.FloatTensor(right_label)
    train_dataset_em.label_update(right_label)

def Initial_E_agg():

    right_label = []
    data_loader = torch.utils.data.DataLoader(dataset=train_dataset_em, batch_size=1000, shuffle=False)
    print("Initial E step =====>")
    for batch_idx, (left_data, right_data, label) in enumerate(tqdm(data_loader)):
        ep = Variable(right_data).float().cuda()
        right_outputs = right_model_agg(ep, label, ep, type=0)
        right_outputs = right_outputs[:, 1].float()
        right_label += list(right_outputs.detach().cpu().numpy())

    right_label = torch.FloatTensor(right_label)
    return right_label

def Initial_E_fix():

    right_label = []
    data_loader = torch.utils.data.DataLoader(dataset=train_dataset_fix, batch_size=1000, shuffle=False)
    print("Initial E step =====>")
    for batch_idx, (left_data, right_data, label, fix_label) in enumerate(tqdm(data_loader)):
        ep = Variable(right_data).float().cuda()
        right_outputs = right_model_fix.forward(ep, fix_label[:, 1], ep, type=0)
        right_outputs = right_outputs[:, 1].float()
        right_label += list(right_outputs.detach().cpu().numpy())

    right_label = torch.FloatTensor(right_label)
    train_dataset_fix.label_update(right_label)


def test(epoch) :

    print('current test epoch = %d' % epoch)

    left_model_em.eval()
    left_model_agg.eval()
    left_model_fix.eval()
    left_model_majority.eval()
    left_model_supervised.eval()

    total_sample = 0
    auc_label = []

    total_corrects_agg = 0
    auc_agg_output = []


    total_corrects_em = 0
    auc_em_output = []

    total_corrects_fix = 0
    auc_fix_output = []

    total_corrects_s = 0
    auc_s_output = []

    total_corrects_m = 0
    auc_m_output = []

    for batch_idx,(images, ep, labels) in enumerate(test_loader):
        images = Variable(images).float().cuda()
        labels = labels.long().cuda()
        auc_label += list(labels)
        total_sample += images.size()[0]


        outputs = left_model_em(images)
        auc_em_output += list(outputs[:, 1].detach().cpu())
        _, predicts = torch.max(outputs.data, 1)
        total_corrects_em += torch.sum(predicts == labels)

        outputs = left_model_agg(images)
        auc_agg_output += list(outputs[:, 1].detach().cpu())
        _, predicts = torch.max(outputs.data, 1)
        total_corrects_agg += torch.sum(predicts == labels)

        outputs = left_model_fix(images)
        auc_fix_output += list(outputs[:, 1].detach().cpu())
        _, predicts = torch.max(outputs.data, 1)
        total_corrects_fix += torch.sum(predicts == labels)

        outputs = left_model_supervised(images)
        auc_s_output += list(outputs[:, 1].detach().cpu())
        _, predicts = torch.max(outputs.data, 1)
        total_corrects_s += torch.sum(predicts == labels)

        outputs = left_model_majority(images)
        auc_m_output += list(outputs[:, 1].detach().cpu())
        _, predicts = torch.max(outputs.data, 1)
        total_corrects_m += torch.sum(predicts == labels)

    auc_em = roc_auc_score(auc_label, auc_em_output)
    acc_em = float(total_corrects_em) / float(total_sample)
    auc_agg = roc_auc_score(auc_label, auc_agg_output)
    acc_agg = float(total_corrects_agg) / float(total_sample)
    auc_fix = roc_auc_score(auc_label, auc_fix_output)
    acc_fix = float(total_corrects_fix) / float(total_sample)
    auc_s = roc_auc_score(auc_label, auc_s_output)
    acc_s = float(total_corrects_s) / float(total_sample)
    auc_m = roc_auc_score(auc_label, auc_m_output)
    acc_m = float(total_corrects_m) / float(total_sample)

    return  acc_em,auc_em,acc_agg,auc_agg,acc_fix,auc_fix,acc_s,auc_s,acc_m,auc_m


if __name__ == '__main__':
    print("True Confusion Matrix:")
    print(confusion_matrix)
    best_agg = 0
    best_em = 0
    best_agg_auc = 0
    best_em_auc = 0
    best_fix = 0
    best_fix_auc = 0
    best_s = 0
    best_s_auc = 0
    best_m = 0
    best_m_auc = 0
    Initial_E_em()
    Initial_E_fix()
    mu_ = Initial_E_agg()
    for epoch in range(Config.epoch_num):

        train_agg(epoch,mu_)
        train_reparameterization()
        train_fix_major()
        train_supervised()
        right_model_fix.print_save_expertise('fix')
        right_model_em.print_save_expertise('em')
        right_model_agg.print_save_expertise()
        acc_em, auc_em, acc_agg, auc_agg, acc_fix, auc_fix, acc_s, auc_s, acc_m, auc_m = test(epoch)


        best_agg, best_agg_auc = max(best_agg,acc_agg), max(best_agg_auc, auc_agg)
        best_em, best_em_auc = max(best_em, acc_em), max(best_em_auc, auc_em)
        best_fix, best_fix_auc = max(best_fix,acc_fix), max(best_fix_auc,auc_fix)
        best_s, best_s_auc = max(best_s, acc_s), max(best_s_auc, auc_s)
        best_m, best_m_auc = max(best_m, acc_m), max(best_m_auc, auc_m)

        print('Learning from crowds Acc:{:.4f}'.format(best_agg))
        print('Our method Acc:{:.4f}'.format(best_em))
        print('Fix majority Acc:{:.4f}'.format(best_fix))
        print('Supervised Acc:{:.4f}'.format(best_s))
        print('Majority Acc:{:.4f}'.format(best_m))

        print('Learning from crowds AUC:{:.4f}'.format(best_agg_auc))
        print('Our method AUC:{:.4f}'.format(best_em_auc))
        print('Fix majority AUC:{:.4f}'.format(best_fix_auc))
        print('Supervised AUC:{:.4f}'.format(best_s_auc))
        print('Majority AUC:{:.4f}'.format(best_m_auc))