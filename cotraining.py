"""
cotraining - main file for training and testing
"""

import torch.nn.functional as F
from model import *
from data import *
from util import *
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import  roc_auc_score
import matplotlib.pyplot as plt
from  common import  args
from tqdm import tqdm
from tensorboardX import SummaryWriter

torch.cuda.set_device(Config.device_id)


def train_fix():

    # Update train loader after refreshing label
    train_loader_fix = torch.utils.data.DataLoader(dataset=train_dataset_fix, batch_size=Config.batch_size, shuffle=True)
    data_loader_fix = torch.utils.data.DataLoader(dataset=train_dataset_fix, batch_size=Config.batch_size, shuffle=False)

    left_model_fix.train()

    for batch_idx, (left_data, right_data, label,fix_label) in enumerate(train_loader_fix):
        ep = Variable(right_data).float().cuda()
        images = Variable(left_data).float().cuda()
        label = label.float().cuda()
        left_optimizer_fix.zero_grad()
        left_outputs = left_model_fix(images)
        right_outputs = right_model_em.forward(ep, label, left_outputs, type=3)
        loss = F.binary_cross_entropy(right_outputs[:, 1].float() + (right_outputs[:, 1] < 0.01).float() * 0.01, label)
        loss.backward()
        left_optimizer_fix.step()

    for iter_num in range(2):
        for batch_idx, (left_data, right_data, label, fix_label) in enumerate(train_loader_fix):
            ep = Variable(right_data).float().cuda()
            images = Variable(left_data).float().cuda()
            label = label.float().cuda()

            left_outputs = left_model_fix(images)
            right_outputs = right_model_fix.forward(ep, fix_label[:,1], left_outputs, type=3)

            loss = F.binary_cross_entropy(right_outputs[:, 1].float() + (right_outputs[:, 1] < 0.01).float() * 0.01,label)
            loss.backward()
            right_model_fix.weights_update()

    print("E step =====>")
    # E-step
    right_label = []
    for batch_idx, (left_data, right_data, label,fix_label) in enumerate(data_loader_fix):
        ep = Variable(right_data).float().cuda()
        images = Variable(left_data).float().cuda()
        left_outputs = left_model_fix(images)
        right_outputs = right_model_fix.forward(ep, fix_label[:,1], left_outputs, type=3)
        right_outputs = right_outputs[:, 1].float()
        right_label += list(right_outputs.detach().cpu().numpy())
    right_label = torch.FloatTensor(right_label)
    if Config.save:
        np.save('./labels/fix_majority_label.npy', right_label.detach().cpu().numpy())
    train_dataset_fix.label_update(right_label)

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
    if Config.save:
        np.save('./labels/em_label.npy', right_label.detach().cpu().numpy())

def train_m():

    # training
    left_model_m.train()
    data_loader_supervised = torch.utils.data.DataLoader(dataset=train_dataset_supervised,batch_size=Config.batch_size, shuffle=False)


    for batch_idx, (left_data, right_data, label) in enumerate(train_loader_supervised):

        images = Variable(left_data).float().cuda()
        ep = Variable(right_data).float().cuda()
        label = torch.max(torch.sum(ep,1),1)[1].float().cuda()

        left_outputs = left_model_m(images)
        loss = F.binary_cross_entropy(left_outputs[:,1].float(), label)
        loss.backward()
        left_optimizer_m.step()

    right_label = []
    correct_label = []
    for batch_idx, (left_data, right_data, label) in enumerate(data_loader_supervised):

        images = Variable(left_data).float().cuda()
        left_outputs = left_model_m(images)
        right_outputs = left_outputs[:, 1].float()
        right_label += list(right_outputs.detach().cpu().numpy())
        correct_label += list(label.detach().cpu().numpy())
    right_label = torch.FloatTensor(right_label)
    correct_label = torch.FloatTensor(correct_label)
    if Config.save:
        np.save('./labels/m_label.npy', right_label.detach().cpu().numpy())
        np.save('./labels/correct_label.npy', correct_label.detach().cpu().numpy())

def train_agg():
    # AggNet

    train_loader_agg = torch.utils.data.DataLoader(dataset=train_dataset_agg, batch_size=Config.batch_size, shuffle=True)
    data_loader_agg = torch.utils.data.DataLoader(dataset=train_dataset_agg, batch_size=Config.batch_size,shuffle=False)
    # training
    left_model_agg.train()
    right_model_agg.train()
    # M-step

    for batch_idx, (left_data, right_data, label) in enumerate(train_loader_agg):
        ep = Variable(right_data).float().cuda()
        images = Variable(left_data).float().cuda()
        label = label.float().cuda()
        left_optimizer_agg.zero_grad()
        left_outputs = left_model_agg(images)
        right_outputs = right_model_agg(ep, 1, left_outputs, type=3)
        loss = F.binary_cross_entropy(right_outputs[:, 1].float() + (right_outputs[:, 1] < 0.01).float() * 0.01, label)
        loss.backward()
        left_optimizer_agg.step()


    right_outputs_all = []
    ep_label_all = []
    for batch_idx, (left_data, right_data, label) in enumerate(data_loader_agg):
        ep = Variable(right_data).float().cuda()
        images = Variable(left_data).float().cuda()
        left_outputs = left_model_agg(images)
        right_outputs = right_model_agg(ep, 1, left_outputs, type=3)
        right_outputs = list(right_outputs.detach().cpu().numpy())
        right_outputs_all += right_outputs

        expert_label = torch.zeros((ep.size()[0], ep.size()[1])).cuda()
        max_element,expert_label[:, :] = torch.max(ep,2)
        mask = (max_element == 0)
        expert_label = expert_label + (-Config.num_classes) * mask.float().cuda()
        expert_label = list(expert_label.detach().cpu().numpy())
        ep_label_all += expert_label

    right_outputs_all = torch.FloatTensor(right_outputs_all)
    ep_label_all = torch.FloatTensor(ep_label_all)
    expert_parameters = M_step(ep_label_all, right_outputs_all)
    right_model_agg.weights_update(expert_parameters)

    # E-step
    right_label = []
    for batch_idx, (left_data, right_data, label) in enumerate(data_loader_agg):
        ep = Variable(right_data).float().cuda()
        images = Variable(left_data).float().cuda()
        left_outputs = left_model_agg(images)
        right_outputs = right_model_agg(ep, 1,left_outputs, type=3)
        right_outputs = right_outputs[:, 1].float()
        right_label += list(right_outputs.detach().cpu().numpy())

    right_label = torch.FloatTensor(right_label)
    if Config.save:
        np.save('./labels/agg_label.npy', right_label.detach().cpu().numpy())
    train_dataset_agg.label_update(right_label)



    return 0

def Initial_E_agg():
    right_label = []
    for batch_idx, (left_data, right_data, label) in enumerate(data_loader_agg):
        ep = Variable(right_data).float().cuda()
        right_outputs = right_model_agg(ep,label, ep, type=0)
        right_outputs = right_outputs[:, 1].float()
        right_label += list(right_outputs.detach().cpu().numpy())
    right_label = torch.FloatTensor(right_label)
    train_dataset_agg.label_update(right_label)

def Initial_E_em():

    right_label = []
    data_loader = torch.utils.data.DataLoader(dataset=train_dataset_em, batch_size=Config.batch_size, shuffle=False)

    for batch_idx, (left_data, right_data, label) in enumerate(data_loader):
        ep = Variable(right_data).float().cuda()
        right_outputs = right_model_em.forward(ep, label, ep, type=0)
        right_outputs = right_outputs[:, 1].float()
        right_label += list(right_outputs.detach().cpu().numpy())
    right_label = torch.FloatTensor(right_label)
    train_dataset_em.label_update(right_label)

def Initial_E_fix():

    right_label = []
    data_loader = torch.utils.data.DataLoader(dataset=train_dataset_fix, batch_size=Config.batch_size, shuffle=False)

    for batch_idx, (left_data, right_data, label,fix_label) in enumerate(data_loader):
        ep = Variable(right_data).float().cuda()
        right_outputs = right_model_fix.forward(ep,fix_label[:,1], ep, type=0)
        right_outputs = right_outputs[:, 1].float()
        right_label += list(right_outputs.detach().cpu().numpy())
    right_label = torch.FloatTensor(right_label)
    train_dataset_fix.label_update(right_label)




def test(epoch) :

    print('current test epoch = %d' % epoch)

    left_model_em.eval()
    left_model_agg.eval()
    left_model_fix.eval()
    left_model_m.eval()

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

        outputs = left_model_m(images)
        auc_m_output += list(outputs[:, 1].detach().cpu())
        _, predicts = torch.max(outputs.data, 1)
        total_corrects_m += torch.sum(predicts == labels)

    auc_em = roc_auc_score(auc_label, auc_em_output)
    acc_em = float(total_corrects_em) / float(total_sample)
    auc_agg = roc_auc_score(auc_label, auc_agg_output)
    acc_agg = float(total_corrects_agg) / float(total_sample)
    auc_fix = roc_auc_score(auc_label, auc_fix_output)
    acc_fix = float(total_corrects_fix) / float(total_sample)
    auc_m = roc_auc_score(auc_label, auc_m_output)
    acc_m = float(total_corrects_m) / float(total_sample)

    return  acc_em,auc_em,acc_agg,auc_agg,acc_fix,auc_fix,acc_m,auc_m



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
    Initial_E_agg()
    for epoch in range(Config.epoch_num):
        train_agg()
        train_em()
        train_fix()
        train_m()
        if Config.save:
            right_model_fix.print_save_expertise('fixdog')
            right_model_em.print_save_expertise('emdog')
            right_model_agg.print_save_expertise()
        acc_em, auc_em, acc_agg, auc_agg, acc_fix, auc_fix, acc_m, auc_m = test(epoch)

        best_agg, best_agg_auc = max(best_agg, acc_agg), max(best_agg_auc, auc_agg)
        best_em, best_em_auc = max(best_em, acc_em), max(best_em_auc, auc_em)
        best_fix, best_fix_auc = max(best_fix, acc_fix), max(best_fix_auc, auc_fix)
        #best_s, best_s_auc = max(best_s, acc_s), max(best_s_auc, auc_s)
        best_m, best_m_auc = max(best_m, acc_m), max(best_m_auc, auc_m)

        print('Learning from crowds Acc:{:.4f}'.format(best_agg))
        print('Our method Acc:{:.4f}'.format(best_em))
        print('Fix majority Acc:{:.4f}'.format(best_fix))
        #print('Supervised Acc:{:.4f}'.format(best_s))
        print('Majority Acc:{:.4f}'.format(best_m))

        print('Learning from crowds AUC:{:.4f}'.format(best_agg_auc))
        print('Our method AUC:{:.4f}'.format(best_em_auc))
        print('Fix majority AUC:{:.4f}'.format(best_fix_auc))
        #print('Supervised AUC:{:.4f}'.format(best_s_auc))
        print('Majority AUC:{:.4f}'.format(best_m_auc))