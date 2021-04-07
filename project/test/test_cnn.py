import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
from tqdm import tqdm
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        ''' n_{out} = floor[(n_{in} + 2p - k) / s] + 1 '''
        # n_{in}: num of input features
        # n_{out}: num of output features
        # k: convolution kernel size
        # s: convolution strode size
        # p: convolution padding size

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

        # weight init
        self.initialize_weights()


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x


    def initialize_weights(self):
        """weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')  # default Leaky relu
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def calc_n_out(n_in: int, k: int, s: int, p: int):
    """
    n_{out} = floor[(n_{in} + 2p - k) / s] + 1
    :param n_in: num of input features
    :param k: convolution kernel size
    :param s: convolution strode size
    :param p: convolution padding size
    :return n_out: num of output features
    """
    return math.floor((n_in + 2 * p - k) / s) + 1


def im_show(img):
    # img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    num_correct = 0
    mum_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            # 64x10
            _, preds = scores.max(1)
            num_correct += torch.sum((preds == y), -1)
            mum_samples += preds.size(0)
        accuracy = float(num_correct) / float(mum_samples)
        print(f"Got {num_correct} / {mum_samples} with accuracy: {accuracy * 100: .2f}%")
    model.train()
    return accuracy


if __name__ == '__main__':
    print(torch.__version__)
    device: torch.device
    # init_gpu(use_gpu=True, gpu_id=0)
    params = {
        "input_size": 1,
        "num_classes": 10,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "num_epochs": 3,
    }

    #######################################################
    ''' sanity check '''
    CHECK = False
    if CHECK:
        print(f"n_out: {calc_n_out(n_in=28, k=3, s=1, p=1)}")
        model = CNN(in_channels=params["input_size"], num_classes=params["num_classes"])
        x = torch.randn(64, 1, 28, 28)
        print(x.shape)
        s = model(x)
        print(s.shape)
        assert s.shape[0] == 64 and s.shape[1] == params["num_classes"]
        del model
    #######################################################

    # load data
    train_dataset = datasets.MNIST(
        root='data/',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )

    test_dataset = datasets.MNIST(
        root="data",
        train=False,
        transform=transforms.ToTensor(),
        download=True,
    )

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=params["batch_size"], shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=params["batch_size"], shuffle=True)

    # visualize on batch
    VISUAL = False
    if VISUAL:
        data_iter = iter(train_dataloader)
        images, labels = data_iter.next()
        im_show(torchvision.utils.make_grid(images))

    # Init GPU if available
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    print(device)

    # Init network
    model = CNN(in_channels=params["input_size"], num_classes=params["num_classes"]).to(device)
    model.train()

    # Init loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

    # Training
    for epoch in tqdm(range(params["num_epochs"])):
        for batch_idx, (data, targets) in enumerate(train_dataloader):
            # transfer data to cuda if available
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            scores = model(data)
            loss = criterion(scores, targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # gradient descent
            optimizer.step()

    # check accuracy
    check_accuracy(train_dataloader, model)
    check_accuracy(test_dataloader, model)
