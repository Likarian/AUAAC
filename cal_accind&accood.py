import torch
import numpy as np
import argparse

import glob
import csv
import os
from os import path as osp

from torchvision import transforms
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torchvision.datasets.folder import default_loader
from PIL import Image
from resnet import ResNet34


np.set_printoptions(suppress=True,
   formatter={'float_kind':'{:0.5f}'.format}) 


class ImageFolderOOD(datasets.VisionDataset):
    def __init__(self, root, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        img_path_list = (
            glob.glob(osp.join(root, "*", "*.jpeg"))
            + glob.glob(osp.join(root, "*", "*.png"))
            + glob.glob(osp.join(root, "*", "*.jpg"))
            + glob.glob(osp.join(root, "*", "*", "*.JPEG"))
            + glob.glob(osp.join(root, "*", "*.JPEG"))
        )

        self.data_paths = img_path_list
        self.targets = [-1] * len(img_path_list)

    def __getitem__(self, index):

        img_path, target = self.data_paths[index], self.targets[index]

        img = np.array(default_loader(img_path))
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data_paths)

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, required=True, help='net type')
parser.add_argument('--dataset', type=str, required=True, help='train dataset')
parser.add_argument('--ood', type=str, required=True, help='ood name')
parser.add_argument('--specific', type=str, required=True, help='specific information')
args = parser.parse_args()

weight = './model/'+args.net+'_'+args.dataset+'_'+args.specific+'.pth'

if args.dataset == 'cifar100':
    model = ResNet34(num_c=100)
else:
    model = ResNet34(num_c=10)

data_transform = transforms.Compose(
            [
                transforms.CenterCrop(size=(32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

if args.dataset == 'cifar10':
    testset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=data_transform
    )
    testloader = data.DataLoader(
        testset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
elif args.dataset == 'cifar100':
    testset = datasets.CIFAR100(
        root='./data', train=False, download=True, transform=data_transform
    )
    testloader = data.DataLoader(
        testset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
else:
    testset = datasets.SVHN(
        root='./data', split="test", download=True, transform=data_transform
    )
    testloader = data.DataLoader(
        testset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


if args.ood == 'cifar10':
    oodtestset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=data_transform
    )
    oodtestloader = data.DataLoader(
        oodtestset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
elif args.ood == 'cifar100':
    oodtestset = datasets.CIFAR100(
        root='./data', train=False, download=True, transform=data_transform
    )
    oodtestloader = data.DataLoader(
        oodtestset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
elif args.ood == 'svhn':
    oodtestset = datasets.SVHN(
        root='./data', split="test", download=True, transform=data_transform
    )
    oodtestloader = data.DataLoader(
        oodtestset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
else:
    oodtestset = ImageFolderOOD(root='./data/'+args.ood, transform=data_transform)
    oodtestloader = torch.utils.data.DataLoader(
        oodtestset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

model.load_state_dict(torch.load(weight))
model.eval()
model = model.cuda()

correct_1 = torch.zeros(102)
acc_ind = torch.zeros(102)
data_num = torch.zeros(102)

oodcorrect_1 = torch.zeros(102)
acc_ood = torch.zeros(102)
ooddata_num = torch.zeros(102)


total_num = 0
oodtotal_num = 0

with torch.no_grad():
    for n_iter, (image, label) in enumerate(testloader):
        image = image.cuda()
        label = label.cuda()

        output = model(image)
        total_num += output.shape[0]
        output = torch.nn.functional.softmax(output, dim=-1)

        values, pred = output.topk(1, largest=True, sorted=True)
        values = values.data.cpu()

        label = label.view(label.size(0), -1).expand_as(pred)
        correct = pred.eq(label).float()

        correct= correct.data.cpu()

        for i in range(102):
            start_prob = 0.0+0.01*i
            target_values_inds = (values >= start_prob)
            target_values_inds = target_values_inds * 1
            target_values_inds = torch.nonzero(target_values_inds)

            correct_1[i] += correct[target_values_inds[:,0]].sum()
            data_num[i] += target_values_inds.shape[0]

    for n_iter, (image, label) in enumerate(oodtestloader):
        image = image.cuda()
        label = label.cuda()

        output = model(image)
        oodtotal_num += output.shape[0]
        output = torch.nn.functional.softmax(output, dim=-1)

        values, pred = output.topk(1, largest=True, sorted=True)
        values = values.data.cpu()

        for i in range(102):
            start_prob = 0.00+0.01*i
            target_values_inds = (values < start_prob)
            target_values_inds = target_values_inds * 1
            target_values_inds = torch.nonzero(target_values_inds)

            oodcorrect_1[i] += target_values_inds.shape[0]

correct_1 = correct_1.data.cpu().numpy()
oodcorrect_1 = oodcorrect_1.data.cpu().numpy()

acc_ind = correct_1 / total_num
acc_ood = oodcorrect_1 / oodtotal_num


with open('./result/'+args.net+'_'+args.dataset+'_'+args.ood+'_'+args.specific+'.csv', 'w') as f:
    writer = csv.writer(f)
                       
    writer.writerow([ 'ACC-IND',acc_ind[0],acc_ind[1],acc_ind[2],acc_ind[3],acc_ind[4],acc_ind[5],acc_ind[6],acc_ind[7],acc_ind[8],acc_ind[9],acc_ind[10],acc_ind[11],acc_ind[12],acc_ind[13],acc_ind[14],acc_ind[15],acc_ind[16],acc_ind[17],acc_ind[18],acc_ind[19]
                           ,acc_ind[20],acc_ind[21],acc_ind[22],acc_ind[23],acc_ind[24],acc_ind[25],acc_ind[26],acc_ind[27],acc_ind[28],acc_ind[29],acc_ind[30],acc_ind[31],acc_ind[32],acc_ind[33],acc_ind[34],acc_ind[35],acc_ind[36],acc_ind[37],acc_ind[38],acc_ind[39]
                           ,acc_ind[40],acc_ind[41],acc_ind[42],acc_ind[43],acc_ind[44],acc_ind[45],acc_ind[46],acc_ind[47],acc_ind[48],acc_ind[49]
                           ,acc_ind[50],acc_ind[51],acc_ind[52],acc_ind[53],acc_ind[54],acc_ind[55],acc_ind[56],acc_ind[57],acc_ind[58],acc_ind[59],acc_ind[60],acc_ind[61],acc_ind[62],acc_ind[63],acc_ind[64],acc_ind[65],acc_ind[66],acc_ind[67],acc_ind[68],acc_ind[69]
                           ,acc_ind[70],acc_ind[71],acc_ind[72],acc_ind[73],acc_ind[74],acc_ind[75],acc_ind[76],acc_ind[77],acc_ind[78],acc_ind[79]
                           ,acc_ind[80],acc_ind[81],acc_ind[82],acc_ind[83],acc_ind[84],acc_ind[85],acc_ind[86],acc_ind[87],acc_ind[88],acc_ind[89],acc_ind[90],acc_ind[91],acc_ind[92],acc_ind[93],acc_ind[94],acc_ind[95],acc_ind[96],acc_ind[97],acc_ind[98],acc_ind[99],acc_ind[100],acc_ind[101]
    ])    

    writer.writerow([ 'ACC-OOD',acc_ood[0],acc_ood[1],acc_ood[2],acc_ood[3],acc_ood[4],acc_ood[5],acc_ood[6],acc_ood[7],acc_ood[8],acc_ood[9],acc_ood[10],acc_ood[11],acc_ood[12],acc_ood[13],acc_ood[14],acc_ood[15],acc_ood[16],acc_ood[17],acc_ood[18],acc_ood[19]
                           ,acc_ood[20],acc_ood[21],acc_ood[22],acc_ood[23],acc_ood[24],acc_ood[25],acc_ood[26],acc_ood[27],acc_ood[28],acc_ood[29],acc_ood[30],acc_ood[31],acc_ood[32],acc_ood[33],acc_ood[34],acc_ood[35],acc_ood[36],acc_ood[37],acc_ood[38],acc_ood[39]
                           ,acc_ood[40],acc_ood[41],acc_ood[42],acc_ood[43],acc_ood[44],acc_ood[45],acc_ood[46],acc_ood[47],acc_ood[48],acc_ood[49]
                           ,acc_ood[50],acc_ood[51],acc_ood[52],acc_ood[53],acc_ood[54],acc_ood[55],acc_ood[56],acc_ood[57],acc_ood[58],acc_ood[59],acc_ood[60],acc_ood[61],acc_ood[62],acc_ood[63],acc_ood[64],acc_ood[65],acc_ood[66],acc_ood[67],acc_ood[68],acc_ood[69]
                           ,acc_ood[70],acc_ood[71],acc_ood[72],acc_ood[73],acc_ood[74],acc_ood[75],acc_ood[76],acc_ood[77],acc_ood[78],acc_ood[79]
                           ,acc_ood[80],acc_ood[81],acc_ood[82],acc_ood[83],acc_ood[84],acc_ood[85],acc_ood[86],acc_ood[87],acc_ood[88],acc_ood[89],acc_ood[90],acc_ood[91],acc_ood[92],acc_ood[93],acc_ood[94],acc_ood[95],acc_ood[96],acc_ood[97],acc_ood[98],acc_ood[99],acc_ood[100],acc_ood[101]
    ]) 


