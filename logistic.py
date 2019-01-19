import torch
import torch.nn as nn
from common import Config
import numpy as np
torch.cuda.set_device(Config.device_id)


class left_neural_net(nn.Module):
    def __init__(self):
        super(left_neural_net, self).__init__()
        self.linear = torch.nn.Linear(3,1)
        self.lr = 1
    def forward(self, x):

        x = self.linear(x)

        return torch.cat((1-torch.sigmoid(x),torch.sigmoid(x)),dim=1)

    def update_weights(self,soft_label,data,left_output):
        # soft_label size : train_size * 1
        # data size : train_size * 3
        # left_output : train_size * 1
        soft_label =  soft_label.unsqueeze(1)
        left_output = left_output.unsqueeze(1)

        g = torch.sum((soft_label - left_output) * data,0).cuda()
        H = -1 * (((left_output) * data).transpose(1,0) @ ((1-left_output) * data)).cuda()

        for name, module in self.named_children():

            module.weight.data = module.weight.data - self.lr  * (torch.inverse(H) @ g)
            print(module.weight.data)

    def weights(self):

        for name, module in self.named_children():

            return module.weight.data

    def weights_(self,w):
        for name, module in self.named_children():

            module.weight.data = w


left_model_em = left_neural_net().cuda()
w = left_model_em.weights()
left_model_agg = left_neural_net().cuda()
left_model_agg.weights_(w)
left_model_fix = left_neural_net().cuda()
left_model_fix.weights_(w)
left_model_supervised = left_neural_net().cuda()
left_model_supervised.weights_(w)
left_model_majority = left_neural_net().cuda()
left_model_majority.weights_(w)
