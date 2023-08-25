import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class client_model_FedReCo(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

        if self.name == 'EMNIST':
            self.n_cls = 10
            self.fc1 = nn.Linear(1 * 28 * 28, 1024)
            self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, self.n_cls)
            self.weight_keys = [['fc1.weight', 'fc1.bias'],
                                ['fc2.weight', 'fc2.bias'],
                                ['fc3.weight', 'fc3.bias']]


        if  self.name == 'FMNIST':
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=1,  out_channels=4, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(in_channels=4,  out_channels=12, kernel_size=5)
            self.fc1 = nn.Linear(12 * 4 * 4, 1024)
            self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, self.n_cls)

            self.weight_keys = [['conv1.weight', 'conv1.bias'],
                                ['conv2.weight', 'conv2.bias'],
                                ['fc1.weight', 'fc1.bias'],
                                ['fc2.weight', 'fc2.bias'],
                                ['fc3.weight', 'fc3.bias']]


        if self.name == 'CIFAR10':
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 5 * 5, 1024)
            self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, self.n_cls)
            self.weight_keys = [['conv1.weight', 'conv1.bias'],
                                ['conv2.weight', 'conv2.bias'],
                                ['fc1.weight', 'fc1.bias'],
                                ['fc2.weight', 'fc2.bias'],
                                ['fc3.weight', 'fc3.bias']]


        if self.name == 'CIFAR100':
            self.n_cls = 100
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 5 * 5, 1024)
            self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, self.n_cls)
            self.weight_keys = [['conv1.weight', 'conv1.bias'],
                                ['conv2.weight', 'conv2.bias'],
                                ['fc1.weight', 'fc1.bias'],
                                ['fc2.weight', 'fc2.bias'],
                                ['fc3.weight', 'fc3.bias']]
        
        print('it is fedreco')


    def forward(self, x):

        if self.name == 'EMNIST':
            x = x.view(-1, 1 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

        if self.name == 'FMNIST':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 12 * 4 * 4)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)


        if self.name == 'CIFAR10':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64 * 5 * 5)
            x = F.relu(self.fc1(x))
            rep = F.relu(self.fc2(x))
            logit = self.fc3(rep)

        if self.name == 'CIFAR100':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)


        return rep, logit
