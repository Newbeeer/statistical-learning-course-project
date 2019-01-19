# model - models for different methods
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import Config
from data import expert_tmatrix
from torchvision import models
import numpy as np
torch.cuda.set_device(Config.device_id)

p_pure = torch.FloatTensor([0.5,0.5])
#p_pure = initial_priori(train_loader)
p = p_pure

class right_agg():

    def __init__(self,expert_num,eps):

        # eps : learning rate
        # weights : before reparameterization
        #

        self.expert_num = expert_num
        self.priority = torch.tensor([[0.5,0.5]]).cuda()
        self.weights = torch.zeros((expert_num,2))
        self.weight_init()

        self.weights = torch.autograd.Variable(self.weights.cuda(),requires_grad=True)
        self.eps = eps

    def forward(self,y,label,left_p,type):
        #
        #   y : batch_size * expert_num * class_num
        #   label : batch_size
        #   left_p : the output of any classifier(nn,logistic,....) , batch_size * class_num
        #
        # NOTE: implicit assumption of class_num = 2


        #Reconstruct confusion matrix

        conf_mat = torch.zeros((self.expert_num, 2, 2)).cuda()
        conf_mat[:, 0, 0] = self.weights[:, 0]
        conf_mat[:, 1, 1] = self.weights[:, 1]
        conf_mat[:, 0, 1], conf_mat[:, 1, 0]= 1 - conf_mat[:, 0, 0], 1 - conf_mat[:, 1, 1]

        conf_mat = torch.log(conf_mat + 0.001) # prevent overflow

        # output size : batch_size * class_num

        output = (conf_mat @ y.unsqueeze(3)).sum(1).squeeze(2)
        if type == 1 :
            output += torch.log(left_p + 0.001) + torch.log(self.priority)
        elif type == 2 :
            output += torch.log(self.priority)
        elif type == 3 :
            output += torch.log(left_p + 0.001)


        return torch.nn.functional.softmax(output,dim=1)

    def weight_init(self):

        # calculate initial \lambda = 4 * log(\alpha / (1-\alhpa))

        self.weights[:, 0] = expert_tmatrix[:,0,0]
        self.weights[:, 1] = expert_tmatrix[:,1,1]


    def weights_update(self, expert_parameters):
        for i in range(Config.expert_num):
            self.weights[i,0] = expert_parameters[i,0,0]
            self.weights[i,1] = expert_parameters[i,1,1]


class right_stat():

    def __init__(self,expert_num,eps):

        # eps : learning rate
        # weights : before reparameterization
        #

        self.expert_num = expert_num
        self.priority = torch.tensor([[0.5,0.5]]).cuda()
        self.weights = torch.zeros((expert_num,2))
        self.weight_init()

        self.weights = torch.autograd.Variable(self.weights.cuda(),requires_grad=True)
        self.eps = eps

    def forward(self,y,label,left_p,type):
        #
        #   y : batch_size * expert_num * class_num
        #   label : batch_size
        #   left_p : the output of any classifier(nn,logistic,....) , batch_size * class_num
        #
        # NOTE: implicit assumption of class_num = 2

        mu = torch.zeros((label.size()[0], 2, 1)).cuda()
        mu[:, 1, 0], mu[:, 0, 0] = label, 1-label
        mu = (mu - 0.5) ** 2

        #Reconstruct confusion matrix

        conf_mat = torch.zeros((y.size()[0], self.expert_num, 2, 2)).cuda()
        conf_mat[:, :, 0, 0] = torch.sigmoid(mu[:, 0] @ self.weights[:, 0].unsqueeze(0))
        conf_mat[:, :, 1, 1] = torch.sigmoid(mu[:, 1] @ self.weights[:, 1].unsqueeze(0))
        conf_mat[:, :, 0, 1], conf_mat[:, :, 1, 0]= 1 - conf_mat[:, :, 0, 0], 1 - conf_mat[:, :, 1, 1]

        new_conf_mat = torch.log(conf_mat + 0.001) # prevent overflow

        # output size : batch_size * class_num
        output = (new_conf_mat @ y.unsqueeze(3)).sum(1).squeeze(2)
        if type == 1 :
            output += torch.log(left_p + 0.001) + torch.log(self.priority)
        elif type == 2 :
            output += torch.log(self.priority)
        elif type == 3 :
            output += torch.log(left_p + 0.001)


        return torch.nn.functional.softmax(output,dim=1)

    def forward_2(self,y,label,left_p,type):
        #
        #   y : batch_size * expert_num * class_num
        #   label : batch_size
        #   left_p : the output of any classifier(nn,logistic,....) , batch_size * class_num
        #
        # NOTE: implicit assumption of class_num = 2


        mu = left_p.unsqueeze(2).clone().cuda()
        mu = (mu - 0.5) ** 2

        #Reconstruct confusion matrix

        conf_mat = torch.zeros((y.size()[0], self.expert_num, 2, 2)).cuda()
        conf_mat[:, :, 0, 0] = torch.sigmoid(mu[:, 0] @ self.weights[:, 0].unsqueeze(0))
        conf_mat[:, :, 1, 1] = torch.sigmoid(mu[:, 1] @ self.weights[:, 1].unsqueeze(0))
        conf_mat[:, :, 0, 1], conf_mat[:, :, 1, 0]= 1 - conf_mat[:, :, 0, 0], 1 - conf_mat[:, :, 1, 1]

        new_conf_mat = torch.log(conf_mat + 0.001) # prevent overflow

        # output size : batch_size * class_num
        output = (new_conf_mat @ y.unsqueeze(3)).sum(1).squeeze(2)
        if type == 1 :
            output += torch.log(left_p + 0.001) + torch.log(self.priority)
        elif type == 2 :
            output += torch.log(self.priority)
        elif type == 3 :
            output += torch.log(left_p + 0.001)


        return torch.nn.functional.softmax(output,dim=1)

    def weight_init(self):

        # calculate initial \lambda = 4 * log(\alpha / (1-\alhpa))

        self.weights[:, 0] = 4 * torch.log(expert_tmatrix[:,0,0] / (1-expert_tmatrix[:,0,0]))
        self.weights[:, 1] = 4 * torch.log(expert_tmatrix[:,1,1] / (1-expert_tmatrix[:,1,1]))


    def weight_zero(self):
        self.weights.grad.data.zero_()

    def weights_update(self):

        #Updating the weights of experts after backward()

        self.weights.data -= self.eps * self.weights.grad.data
        self.weights.grad.data.zero_()

        self.weights.data[:, 1] = self.weights.data[:, 1].clamp(0, 100)
        self.weights.data[:, 0] = self.weights.data[:, 0].clamp(0, 100)

    def print_save_expertise(self,str):
        np.save('./expertise_'+str,self.weights.detach().cpu().numpy())



class left_neural_net(nn.Module):
    def __init__(self):
        super(left_neural_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv1_batch_norm = nn.BatchNorm2d(32)
        self.classifier  = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, int(Config.num_classes)),
        )

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = self.conv1_batch_norm(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = x.view(-1, 64 * 7 * 7)
        x = self.classifier(x)

        return torch.nn.functional.softmax(x,dim=1)




class right_neural_net_EM(nn.Module):

    def __init__(self,expert_num,copy_expert=0):

        super(right_neural_net_EM, self).__init__()
        self.priority = p_pure.cuda()
        self.p = nn.Linear(1,2,bias=False)

        for i in range(Config.expert_num):
            m_name = "fc" + str(i + 1)
            self.add_module(m_name,nn.Linear(Config.num_classes, Config.num_classes, bias=False))
        self.weights_init()

    def forward(self, x, label, left_p, type=0) :

        #entity = torch.ones((x.size()[0],1)).cuda()
        out = 0
        for name, module in self.named_children():
            if name == 'p':
                continue
            index = int(name[2:])

            out += module(x[:, index-1, :])

        #priority =  self.p(entity)
        if type == 1 :
            out += torch.log(left_p+0.001) + torch.log(self.priority)
        elif type == 2 :
            out += torch.log(self.priority)
        elif type == 3 :
            out += torch.log(left_p + 0.001)
        return torch.nn.functional.softmax(out,dim=1)

    def weights_init(self):
        for name, module in self.named_children():
            if name == 'p':
                #module.weight.data = self.priority
                continue
            elif name[0:2] =='fc':
                index = int(name[2:])
                module.weight.data = torch.log(expert_tmatrix[index - 1] + 0.0001).cuda()


    def weights_update(self, expert_parameters):
        for name, module in self.named_children():
            if name == 'p':
                continue
            index = int(name[2:])
            module.weight.data = torch.log(expert_parameters[index - 1] + 0.0001)

    def weights_init_indentity(self):
        for name, module in self.named_children():
            if name == 'p':
                module.weight.data = self.priority
                continue
            module.weight.data = (torch.FloatTensor(np.ones(Config.num_classes)*(-5)+5*np.eye(Config.num_classes))).cuda()

    def get_prior(self):
        for name, module in self.named_children():
            if name == 'p':
                return module.weight.data
    def print_save_expertise(self):
        weight = np.zeros((Config.expert_num,2,2))
        for name, module in self.named_children():
            if name == 'p':
                continue
            index = int(name[2:])
            weight[index-1] = module.weight.data
        np.save('./expertise_agg',weight)

# models and optimizers for different methods
left_model_em = left_neural_net().cuda()
left_model_fix = left_neural_net().cuda()
left_model_m = left_neural_net().cuda()
left_model_agg = left_neural_net().cuda()

right_model_em = right_stat(Config.expert_num,Config.right_learning_rate)
right_model_fix = right_stat(Config.expert_num,Config.right_learning_rate)
right_model_agg = right_neural_net_EM(Config.expert_num,Config.right_learning_rate).cuda()

left_optimizer_em = torch.optim.Adam(left_model_em.parameters(), lr=Config.left_learning_rate)
left_optimizer_agg = torch.optim.Adam(left_model_agg.parameters(), lr = Config.left_learning_rate)
left_optimizer_fix = torch.optim.Adam(left_model_fix.parameters(), lr = Config.left_learning_rate)
left_optimizer_m = torch.optim.Adam(left_model_m.parameters(), lr = Config.left_learning_rate)
