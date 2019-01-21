"""
data - to generate data from crowds
"""
import numpy as np
import torch
import torch.utils
import os
from common import Config
from common import  args
from tqdm import tqdm
from data_generator import mu,datapoint_1,tdatapoint_1,label_train,label_test

torch.set_num_threads(16)

workers = 16

class Im_EP(torch.utils.data.Dataset):
    """
    Im_EP - to generate a dataset with images, experts' predictions and labels for learning from crowds settings
    """
    def __init__(self, as_expertise, root_path, missing_label, train, fix=False):
        self.as_expertise = as_expertise
        self.class_num = Config.num_classes
        self.expert_num = Config.expert_num
        self.root_path = root_path
        self.train = train
        self.missing_label = missing_label
        self.fix = fix
        if self.train:
            self.mu = mu
            self.left_data, self.label = datapoint_1, label_train
            self.right_data = self.generate_data()
            self.fix_major_label = self.major_label_initial()
        else:

            self.mu = mu[:1000]
            self.left_data,self.label = tdatapoint_1, label_test
            self.right_data = self.generate_data()
            self.fix_major_label = self.major_label_initial()

    def __getitem__(self, index):
        if not self.fix:
            if self.train:
                left, right, label = self.left_data[index], self.right_data[index], self.label[index]
            else:
                left, right, label = self.left_data[index], self.right_data[index], self.label[index]
            return left, right, label
        else:
            if self.train:
                left, right, label, major_label = self.left_data[index], self.right_data[index], self.label[index], self.fix_major_label[index]
            else:
                left, right, label, major_label = self.left_data[index], self.right_data[index], self.label[index], self.fix_major_label[index]
            return left, right, label, major_label

    def __len__(self):
        if self.train:
            return 1000
        else:
            return 1000

    def generate_data(self):
        if self.train:
            np.random.seed(1234)
        else:
            np.random.seed(4321)

        ep = np.zeros((self.__len__(), self.expert_num), dtype=np.int)
        labels = np.zeros((self.__len__()), dtype=np.int16)
        right_data = np.zeros((self.__len__(), self.expert_num, self.class_num), dtype=np.float)

        for i in range(self.__len__()):

            labels[i] = self.label[i]

            #Case 1: Our method
            if Config.experiment_case == 1:
                for expert in range(self.expert_num):

                    if labels[i] == 0:
                        prob = 1 / (1 + np.exp(-Config.as_expertise_lambda[expert][0] * (self.mu[i]-0.5)**2))
                        ep[i][expert] = int(np.random.choice(Config.num_classes, 1, p=[prob, 1-prob]))

                    if labels[i] == 1:
                        prob = 1 / (1 + np.exp(-Config.as_expertise_lambda[expert][1] * (self.mu[i]-0.5)**2))
                        ep[i][expert] = int(np.random.choice(Config.num_classes, 1, p=[1-prob, prob]))

                    right_data[i][expert][ep[i][expert]] = 1

            #Case 2:
            if Config.experiment_case == 2:

                for expert in range(self.expert_num):

                    ep[i][expert] = int(np.random.choice(Config.num_classes, 1, p=self.as_expertise[expert][labels[i]]))
                    right_data[i][expert][ep[i][expert]] = 1

            if Config.experiment_case == 3:
                for expert in range(self.expert_num):

                    if labels[i] == 0:
                        prob = 1 / (3 + int(np.random.choice(5,1)))
                        ep[i][expert] = int(np.random.choice(Config.num_classes, 1, p=[1-prob, prob]))

                    if labels[i] == 1:
                        prob = 1 / (3 + int(np.random.choice(5, 1)))
                        #prob = 1 / (1 + np.exp(-Config.as_expertise_lambda[expert][1] * (self.mu[i]-0.5)**2))
                        ep[i][expert] = int(np.random.choice(Config.num_classes, 1, p=[prob, 1-prob]))

                    right_data[i][expert][ep[i][expert]] = 1

        return right_data

    def label_initial(self):
        linear_sum = torch.sum(torch.tensor(self.right_data), dim=1)
        _, major_label = torch.max(linear_sum, 1)
        self.label = major_label

    def major_label_initial(self):
        linear_sum = torch.sum(torch.tensor(self.right_data), dim=1)
        return linear_sum/Config.expert_num

    def label_update(self, new_label):
        self.label = new_label

    def get_expert_label(self):
        return  self.right_data

def Initial_mats():

    if not Config.missing :
        print("Generating inital conf matrix")
        sum_majority_prob = torch.zeros((Config.num_classes))
        confusion_matrix = torch.zeros((Config.expert_num, Config.num_classes, Config.num_classes))
        expert_tmatrix = torch.zeros((Config.expert_num, Config.num_classes, Config.num_classes))

        for i, (img, ep, label) in enumerate(tqdm(train_loader_em)):
            linear_sum = torch.sum(ep, dim=1)
            label = label.long()
            prob = linear_sum / Config.expert_num
            sum_majority_prob += torch.sum(prob, dim=0).float()

            for j in range(ep.size()[0]):
                _, expert_class = torch.max(ep[j], 1)
                linear_sum_2 = torch.sum(ep[j], dim=0)
                prob_2 = linear_sum_2 / Config.expert_num
                for R in range(Config.expert_num):
                    expert_tmatrix[R, :, expert_class[R]] += prob_2.float()
                    confusion_matrix[R, label[j], expert_class[R]] += 1

        for R in range(Config.expert_num):
            linear_sum = torch.sum(confusion_matrix[R, :, :], dim=1)
            confusion_matrix[R, :, :] /= linear_sum.unsqueeze(1)

        expert_tmatrix = expert_tmatrix / sum_majority_prob.unsqueeze(1)
        print("Finish")
        return confusion_matrix, expert_tmatrix
    else:
        sum_majority_prob = torch.zeros((Config.expert_num,Config.num_classes))
        confusion_matrix = torch.zeros((Config.expert_num, Config.num_classes, Config.num_classes))
        expert_tmatrix = torch.zeros((Config.expert_num, Config.num_classes, Config.num_classes))

        for i, (img, ep, label) in enumerate(train_loader_em):
            label = label.long()
            for j in range(ep.size()[0]):
                linear_sum_2 = torch.sum(ep[j], dim=0)
                prob_2 = linear_sum_2 / torch.sum(linear_sum_2)
                for R in range(Config.expert_num):
                    # If missing ....
                    if max(ep[j,R]) == 0:
                        continue
                    _,expert_class = torch.max(ep[j,R],0)
                    expert_tmatrix[R, :, expert_class] += prob_2.float()
                    confusion_matrix[R, label[j], expert_class] += 1
                    sum_majority_prob[R] += prob_2.float()

        for R in range(Config.expert_num):
            linear_sum = torch.sum(confusion_matrix[R], dim=1)
            confusion_matrix[R, :, :] /= linear_sum.unsqueeze(1)
            expert_tmatrix[R] = expert_tmatrix[R] / sum_majority_prob[R].unsqueeze(1)

        return confusion_matrix, expert_tmatrix



# datasets for training and testing
train_dataset_em = Im_EP(as_expertise=Config.as_expertise, root_path=Config.data_root, missing_label=Config.missing_label, train=True)
train_dataset_supervised = Im_EP(as_expertise=Config.as_expertise, root_path=Config.data_root, missing_label=Config.missing_label, train=True)
train_dataset_fix = Im_EP(as_expertise=Config.as_expertise, root_path=Config.data_root, missing_label=Config.missing_label, train=True, fix=True)
train_dataset_fix.label_initial()
train_dataset_em.label_initial()
train_loader_em = torch.utils.data.DataLoader(dataset = train_dataset_em, num_workers = workers,batch_size = 1000, shuffle = True)
test_dataset = Im_EP(as_expertise=Config.as_expertise, root_path=Config.data_root, missing_label=Config.missing_label, train=False)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,num_workers = workers, batch_size = 1000, shuffle = False)
train_dataset_agg = Im_EP(as_expertise=Config.as_expertise, root_path=Config.data_root,missing_label=Config.missing_label, train=True)
train_dataset_agg.label_initial()
data_loader_agg = torch.utils.data.DataLoader(dataset=train_dataset_agg, num_workers = workers,batch_size=1000, shuffle=False)
data_loader_supervised = torch.utils.data.DataLoader(dataset=train_dataset_supervised, num_workers = workers,batch_size=1000, shuffle=False)
#train_dataset_em = Im_EP(as_expertise=Config.as_expertise, root_path=Config.data_root, missing_label=Config.missing_label, train=True)
#train_dataset_em.label_initial()
confusion_matrix, expert_tmatrix = Initial_mats()