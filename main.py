from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from data import *
from resnet_drop_12_09 import *
from Config import Config,args
from util import GetLayer


torch.cuda.set_device(Config.device_id)

def train(model, train_loader, optimizer, epoch):

    model.train()
    correct = 0
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        # remember to set target to be dtype = long

        data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)

        loss = F.nll_loss(output, target)
        loss.backward(retain_graph=False)
        optimizer.step()
        predict = torch.max(output,1)[1]
        correct += predict.eq(target.view_as(predict)).sum().item()

        '''
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        '''

    acc = 100. * correct / len(train_loader.dataset)
    print('\nAccuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(train_loader.dataset),
        acc))

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target,size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        acc))
    return acc

def main():

    model = Net
    optimizer = Net_Optimizer
    best_acc = 0
    for epoch in range(Config.epoch_num):
        print("Current epoch : %d"%(epoch))
        train(model, train_data_loader, optimizer, epoch)
        acc = test(model, test_data_loader)

        if best_acc < acc:

            torch.save({'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'bestacc': best_acc, }, args.checkpoint)
            best_acc = acc



if __name__ == '__main__':
    main()
