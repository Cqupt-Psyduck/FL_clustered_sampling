import os
import pickle

import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# GPU settings
from torchvision import datasets

torch.backends.cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


# [DONE] test finished
# return x_aux: (320, 3, 32, 32)
# y_aux:
# array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
#        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4,
#        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
#        4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
#        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6,
#        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
#        6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
#        7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8,
#        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
#        8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
#        9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9])
###
###
# extract 10 * 32 data from testset, for 10 classes with each class of 32 examples
# here I don't seperate axuiliary dataset from testset due to its small scale
def create_auxiliary_data(dataset, data_size=32):
    ''' return the numpy array of auxiliary data, x: (320, 3, 32, 32), y: (320, ) (class labels from 0 to 9 sorted)'''
    if dataset[:5] == "MNIST":
        # 将测试集的样本拿出来给x，y
        MNIST_test = datasets.MNIST(root="./data", train=False, download=True)
        x, y = MNIST_test.data, MNIST_test.targets
    elif dataset[:5] == "CIFAR":
        CIFAR10_test = datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )
        x, y = CIFAR10_test.data, CIFAR10_test.targets

    x_aux, y_aux = [], []

    # 每个类别提取32个样本
    for i in range(10):
        for (sample, label) in zip(x, y):
            if (label == i) and (len(x_aux) + 1 <= data_size * (i + 1)):
                # x_aux.append(sample.numpy())
                # y_aux.append(label.numpy())
                x_aux.append(np.asarray(sample))
                y_aux.append(np.asarray(label))

    x_aux = np.array(x_aux)
    y_aux = np.array(y_aux)

    folder = "./data/"
    if dataset[:5] == "MNIST":
        auxiliary_path = f"MNIST_auxiliary_data_{data_size}sample_per_class.pkl"
        with open(folder + auxiliary_path, "wb") as output:
            pickle.dump((x_aux, y_aux), output)
    elif dataset[:5] == "CIFAR":
        auxiliary_path = f"CIFAR_auxiliary_data_{data_size}sample_per_class.pkl"
        with open(folder + auxiliary_path, "wb") as output:
            pickle.dump((x_aux, y_aux), output)


class MnistAuxiliaryDataset(Dataset):
    """Convert the MNIST pkl file into a Pytorch Dataset"""

    def __init__(self, file_path):
        with open(file_path, "rb") as pickle_file:
            dataset = pickle.load(pickle_file)
            # 将数组堆叠(10, 32, 32, 3)变为(320, 32, 3)
            # self.features = np.vstack(dataset[0][k])

            self.features = dataset[0]
            # vector_labels = list()
            # for idx, digit in enumerate(dataset[1][k]):
            #     vector_labels += [digit] * len(dataset[0][k][idx])

            self.labels = dataset[1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 3D input 1x28x28
        x = torch.Tensor([self.features[idx]]) / 255
        y = torch.LongTensor([self.labels[idx]])[0]

        return x, y


class CIFARDataset(Dataset):
    """Convert the CIFAR pkl file into a Pytorch Dataset"""

    def __init__(self, file_path: str):

        dataset = pickle.load(open(file_path, "rb"))

        self.X = dataset[0]
        self.y = np.array(dataset[1])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):

        # 3D input 32x32x3
        # permute(2, 0, 1)将(W,H,B)交换维度为(B,W,H)，因为torch里面第一个维度普遍是batchnum，我猜将颜色通道放到第一维度，正好每种颜色通道一个卷积核
        x = torch.Tensor(self.X[idx]).permute(2, 0, 1) / 255
        x = (x - 0.5) / 0.5
        y = self.y[idx]

        return x, y


def get_auxiliary_data_loader(dataset, data_size=32):
    path_train = "./data/" + f"{dataset[:5]}_auxiliary_data_{data_size}sample_per_class.pkl"
    if not os.path.isfile(path_train):
        create_auxiliary_data(dataset, data_size)

    # No matter you choose trainset or testset to generate auxiliary data
    # use the transforms on testdata
    # 我们不使用这个
    # _, transforms = get_default_data_transforms(verbose=False)

    # batch_size = data_size, don't shuffle
    if dataset[:5] == "MNIST":
        auxiliary_data_loader = torch.utils.data.DataLoader(
            MnistAuxiliaryDataset(path_train),
            batch_size=data_size, shuffle=False
        )
    elif dataset[:5] == "CIFAR":
        auxiliary_data_loader = torch.utils.data.DataLoader(
            CIFARDataset(path_train),
            batch_size=data_size, shuffle=False
        )
    else:
        return
    return auxiliary_data_loader


def compute_grad_aux(global_model, aux_loader):
    '''our own version of ratio'''
    optimizer = optim.SGD(global_model.parameters(), lr=0.1)  # lr doesn't matter here

    grad_square_sum_lst = [0] * 10

    # 一次输入一类样本训练模型，每类32张
    for class_i_data_idx, (input, target) in enumerate(aux_loader):

        # 6. zero gradient, otherwise accumulated
        # 梯度置0
        optimizer.zero_grad()

        # 1. prepare (X, Y) which belongs to same class
        # for ith class's data, with shape of [32, 3, 32, 32] batch size pf 32, with image 3*32*32
        # for ith class'labels, don't need one-hot coding here
        input, target = input.to(device), target.type(torch.LongTensor).to(device)

        # 2. forward pass get Y_out
        output = global_model(input)  # feed ith class data, network output with shape [32, 10]
        # the output is logits (before softmax), usually output logits as default

        # 3. calculate cross-entropy between Y and Y_out
        loss = F.cross_entropy(output, target)

        # 4. backward pass to get gradients wrt weight using a batch of 32 of data
        loss.backward()

        # 5. record gradients wrt weights from the last layer
        # print(cifarnet.fc2.weight.grad.shape) here is [500, 10]
        # here we only need fetch ith gradient with shape [500]
        # In short, send data from ith class, fetch ith gradient tensor from [500, 10]
        # grad_lst.append(global_model.fc2.weight.grad[class_i_data_idx].cpu().numpy())
        # new
        # the above descripting all wrong
        # named_parameters()获取神经网络每一层的名字和参数
        for name, param in global_model.named_parameters():
            # print(name, param.grad.shape)
            # print((param.grad ** 2).sum().item())
            # 计算每一层参数的L2范数的平方
            grad_square_sum_lst[class_i_data_idx] += (param.grad ** 2).mean().item()

    return grad_square_sum_lst


# threshold changed the meaning to percentage
# threshold doesn't work here
# grad_square_sum_lst: [10, ]
def compute_ratio(grad_square_sum_lst, temp=1):
    ''' original version in the paper '''

    grad_sum = np.array(grad_square_sum_lst)
    # print(grad_sum)

    grad_sum = grad_sum.min() / grad_sum
    # print(grad_sum)

    # def softmax(grad_sum, temp = 1):
    #     grad_sum = grad_sum - grad_sum.mean()
    #     return np.exp(grad_sum / temp) / np.exp(grad_sum / temp).sum()

    # grad_sum_normalize = softmax(grad_sum, temp)
    grad_sum_normalize = grad_sum / grad_sum.sum()
    # grad_sum_normalize = grad_sum

    return grad_sum_normalize


def compute_ratio_per_client_update(client_models, client_idx, aux_loader):
    ra_dict = []
    for i, client_model_update in enumerate(client_models):
        grad_square_sum_lst = compute_grad_aux(client_model_update, aux_loader)
        grad_sum_normalize = compute_ratio(grad_square_sum_lst)
        ra_dict.append(grad_sum_normalize)

    return ra_dict
# if __name__ == '__main__':
#     get_auxiliary_data_loader('CIFAR')
#     # create_auxiliary_data('MNIST')