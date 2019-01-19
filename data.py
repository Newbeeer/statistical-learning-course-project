"""
data - to generate data from crowds
"""
import numpy as np
import torch
import torch.utils
from PIL import Image
import os
from common import Config
from common import  args
from util import multual_information
from tqdm import tqdm

torch.set_num_threads(16)

workers = 16

class Image_(torch.utils.data.Dataset):
    """
    Image_ - to generate a dataset with images and labels from .csv
    """
    def __init__(self, root, img_transform, train):
        self.root = root
        if train :
            flist = os.path.join(root, "train_file.csv")
        else :
            flist = os.path.join(root, "val_file.csv")
        self.imlist = self.flist_reader(flist)
        self.transform = img_transform
        self.train = train

    def __getitem__(self, index):
        impath, target = self.imlist[index]

        img = Image.open(impath).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imlist)

    def flist_reader(self, flist):
        imlist = []
        #st = '/data1/xuyilun'
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                row = line.split(",")
                impath =  '.' + row[0]
                imlabel = row[1]
                imlist.append((impath, int(imlabel)))
        return imlist


class Im_EP(torch.utils.data.Dataset):
    """
    Im_EP - to generate a dataset with images, experts' predictions and labels for learning from crowds settings
    """
    def __init__(self, as_expertise, root_path, missing_label, train,fix=False):
        self.as_expertise = as_expertise
        self.class_num = Config.num_classes
        self.expert_num = Config.expert_num
        self.root_path = root_path
        self.train = train
        self.fix = fix
        self.missing_label = missing_label
        if self.train:
            train_dataset = Image_(root=self.root_path, img_transform=Config.train_transform, train=True)
            self.mu = self.generate_mu(train_dataset)
            self.left_data, self.right_data, self.label = self.generate_data(train_dataset)
            self.fix_major_label = self.major_label_initial()
        else:
            test_dataset = Image_(root=self.root_path, img_transform=Config.test_transform, train=False)
            self.mu = self.generate_mu(test_dataset)
            self.left_data, self.right_data, self.label = self.generate_data(test_dataset)
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
            return Config.training_size
        else:
            return Config.test_size

    def generate_mu(self,dataset):

        '''
        np.random.seed(1234)
        mu = np.zeros((self.__len__()))

        data_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers=workers, batch_size=1, shuffle=False)
        for i,(img,label) in enumerate(data_loader):
            mu[i] = label + (-2*label+1)  * min(abs(np.random.randn()*0.1),0.5)
        '''

        mu = np.load('mu_1.npy')
        return mu
    def generate_data(self, dataset):
        if self.train:
            np.random.seed(1234)
        else:
            np.random.seed(4321)

        data_loader = torch.utils.data.DataLoader(dataset=dataset, num_workers = workers, batch_size=1, shuffle=False)

        missing_seed = np.random.random((self.__len__(), self.expert_num))
        ep = np.zeros((self.__len__(), self.expert_num), dtype=np.int)
        labels = np.zeros((self.__len__()), dtype=np.int16)
        left_data = np.zeros((self.__len__(), 3, 150, 150))
        right_data = np.zeros((self.__len__(), self.expert_num, self.class_num), dtype=np.float)

        for i, data in enumerate(data_loader):
            left_data[i] = data[0]
            labels[i] = data[1]

            #Case 1: Independent case: 10 experts independently label
            if Config.experiment_case == 1:
                for expert in range(self.expert_num):
                    if Config.missing and missing_seed[i][expert] < self.missing_label[expert]:
                        continue
                    if labels[i] == 0:
                        prob = 1 / (1 + np.exp(-Config.as_expertise_lambda[expert][0] * (self.mu[i]-0.5)**2))
                        ep[i][expert] = int(np.random.choice(Config.num_classes, 1, p=[prob, 1-prob]))

                    if labels[i] == 1:
                        prob = 1 / (1 + np.exp(-Config.as_expertise_lambda[expert][1] * (self.mu[i]-0.5)**2))
                        ep[i][expert] = int(np.random.choice(Config.num_classes, 1, p=[1-prob, prob]))

                    right_data[i][expert][ep[i][expert]] = 1

            #Case 2: 10 normal experts, the other 15 experts always label 0
            #high :5 normal experts/5 the others
            if Config.experiment_case == 2:

                for expert in range(self.expert_num):
                    if Config.missing and missing_seed[i][expert] < self.missing_label[expert]:
                        continue

                    ep[i][expert] = int(np.random.choice(Config.num_classes, 1, p=self.as_expertise[expert][labels[i]]))
                    right_data[i][expert][ep[i][expert]] = 1

            #Case 3: 10 big experts, each experts have 2 small experts seperately.  Attention! Cifar should change!
            if Config.experiment_case == 3:

                for expert in range(Config.senior_num):
                    if Config.missing and missing_seed[i][expert] < self.missing_label[expert]:
                        continue
                    ep[i][expert] = int(np.random.choice(Config.num_classes, 1, p=self.as_expertise[expert][labels[i]]))
                    right_data[i][expert][ep[i][expert]] = 1
                if args.expertise == 0:
                    #5 little experts
                    for expert in range(Config.senior_num,Config.expert_num):
                        #ep[i][expert] = int(np.random.choice(Config.num_classes, 1, p=self.as_expertise[expert][ep[i][0]]))
                        ep[i][expert] = ep[i][0]
                        right_data[i][expert][ep[i][expert]] = 1

                elif args.expertise == 1:
                    for expert in range(Config.senior_num,Config.expert_num):
                        #ep[i][expert] = int(np.random.choice(Config.num_classes, 1, p=self.as_expertise[expert][ep[i][0]]))
                        ep[i][expert] = ep[i][0]
                        right_data[i][expert][ep[i][expert]] = 1


            #Case 4: 5 normal experts, other 5 experts rely on the majority vote of some of experts in the former.
            if Config.experiment_case == 4:
                for expert in range(5):
                    if Config.missing and missing_seed[i][expert] < self.missing_label[expert]:
                        continue
                    ep[i][expert] = int(np.random.choice(Config.num_classes, 1, p=self.as_expertise[expert][labels[i]]))
                    right_data[i][expert][ep[i][expert]] = 1
                for expert in range(5,Config.expert_num):
                    linear_sum = torch.sum(torch.tensor(right_data[i][min((expert-5)%5,(expert-2)%5):max((expert-5)%5,(expert-2)%5)]),dim=0)
                    _, major_label = torch.max(linear_sum, 0)
                    right_data[i][expert][int(major_label)] = 1


        return left_data, right_data, labels

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

def Initial_distribution():
    joint_distr = torch.zeros((1,int(np.power(Config.num_classes,Config.expert_num))))

    for i, (img, ep, label) in enumerate(train_loader_em):
        _, ep = torch.max(ep,2)
        for j in range(len(label)):
            base = 1
            num = ep[j,0]
            for ex in range(1,Config.expert_num):
                num += np.power(2,base)*ep[j,ex]
                base += 1
            joint_distr[0,num] += 1
    joint_distr /= Config.training_size
    return joint_distr



# datasets for training and testing
train_dataset_supervised = Im_EP(as_expertise=Config.as_expertise,root_path=Config.data_root, missing_label=Config.missing_label, train=True)
train_dataset_em = Im_EP(as_expertise=Config.as_expertise, root_path=Config.data_root, missing_label=Config.missing_label, train=True)
train_dataset_em.label_initial()
train_dataset_fix = Im_EP(as_expertise=Config.as_expertise, root_path=Config.data_root, missing_label=Config.missing_label, train=True,fix=True)
train_dataset_fix.label_initial()

train_loader_supervised = torch.utils.data.DataLoader(dataset = train_dataset_supervised, num_workers = workers,batch_size = Config.batch_size, shuffle = True)
train_loader_em = torch.utils.data.DataLoader(dataset = train_dataset_em, num_workers = workers,batch_size = Config.batch_size, shuffle = True)
train_loader_fix = torch.utils.data.DataLoader(dataset = train_dataset_fix, num_workers = workers,batch_size = Config.batch_size, shuffle = True)

test_dataset = Im_EP(as_expertise=Config.as_expertise, root_path=Config.data_root, missing_label=Config.missing_label, train=False)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,num_workers = workers, batch_size = Config.batch_size, shuffle = False)
train_dataset_agg = Im_EP(as_expertise=Config.as_expertise, root_path=Config.data_root,missing_label=Config.missing_label, train=True)
train_dataset_agg.label_initial()
data_loader_agg = torch.utils.data.DataLoader(dataset=train_dataset_agg, num_workers = workers,batch_size=Config.batch_size, shuffle=False)
#train_dataset_em = Im_EP(as_expertise=Config.as_expertise, root_path=Config.data_root, missing_label=Config.missing_label, train=True)
#train_dataset_em.label_initial()
confusion_matrix, expert_tmatrix = Initial_mats()