"""
common - configurations
"""
import torch
import numpy as np
from torchvision.transforms import transforms
import argparse
parser = argparse.ArgumentParser(description='CoTraining')
parser.add_argument('--case', type=int, metavar='N',
                    help='case')
parser.add_argument('--expert_num', type=int, metavar='N',
                    help='case')
parser.add_argument('--device', type=int, metavar='N',
                    help='case')
parser.add_argument('--expertise', type=int, metavar='N',
                    help='case')
args = parser.parse_args()

class Config:
    data_root = '../dogdata'
    #data_root  = '/data1/xuyilun/LUNA16/data'
    training_size = 12500
    test_size = 12500
    #training_size = 6484
    #test_size = 1622
    #as_expertise = np.array([[0.6, 0.8, 0.7, 0.6, 0.7],[0.6,0.6,0.7,0.9,0.6]])
    #-----------------------------


    lexpert = [[0.6,0.4],[0.4,0.6]]
    hexpert = [[.8,.2],[.2,.8]]
    if args.expertise == 0:
        as_expertise = np.array([lexpert,lexpert,lexpert,lexpert,lexpert])
        senior_num = 5
    elif args.expertise == 1:
        '''
        as_expertise = np.array(
            [[[0.9,0.1],[0.1,0.9]], [[0.8,0.2],[0.2,0.8]], [[0.6,0.4],[0.4,0.6]], [[0.7,0.3],[0.3,0.7]], [[0.7,0.3],[0.3,0.7]]])
        '''
        as_expertise = np.array(
            [[[0.6, 0.4], [0.2, 0.8]], [[0.7, 0.3], [0.4, 0.6]], [[0.6, 0.4], [0.4, 0.6]], [[0.7, 0.3], [0.3, 0.7]], [[0.7, 0.3], [0.4, 0.6]]])


        as_expertise_lambda = np.zeros((as_expertise.shape[0],2))
        for i in range(as_expertise.shape[0]):
            as_expertise_lambda[i][0] = 4 * np.log(as_expertise[i][0][0] / (1 - as_expertise[i][0][0]))
            as_expertise_lambda[i][1] = 4 * np.log(as_expertise[i][1][1] / (1 - as_expertise[i][1][1]))


        senior_num = 5



    missing_label = np.array([0, 0, 0, 0, 0])
    missing = False
    num_classes = 2
    left_input_size = 28 * 28
    batch_size = 16
    left_learning_rate = 1e-4
    right_learning_rate = 1e-4
    epoch_num = 30

    #########################

    expert_num = args.expert_num
    device_id = args.device
    experiment_case = args.case

    #########################

    train_transform = transforms.Compose([
            transforms.Resize((150, 150),interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(45),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         ])
    test_transform = transforms.Compose([
            transforms.Resize((150, 150),interpolation=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
         ])
