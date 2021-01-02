import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dataloader import EmotionClassificationDataset
batch_size = 32
use_gpu = torch.cuda.is_available()


def cal_acc(pred, label):
    r = 0
    t = 0
    pred = pred.argmax(dim=1)
    for ii in range(pred.shape[0]):
        t += 1
        if pred[ii] == label[ii]:
            r += 1
    return r, t


class StdModel(nn.Module):
    def __init__(self, num_class):
        self.num_class = num_class
        super(StdModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=(0, 0))  # (24,24)
        self.Relu_1 = torch.relu
        self.maxpooling_1 = nn.MaxPool2d(kernel_size=2, stride=2)  # (12,12)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=(0, 0))  # (10,10)
        self.Relu_2 = torch.relu

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1, padding=(0, 0))  # (10,10) #降维
        self.Relu_3 = torch.relu
        self.maxpooling_3 = nn.MaxPool2d(stride=2, kernel_size=2)  # (5,5)

        self.fc_1 = nn.Linear(in_features=1600, out_features=1000)
        self.sigmoid_1 = torch.sigmoid
        self.fc_2 = nn.Linear(in_features=1000, out_features=num_class)
        self.softmax = F.softmax

    def forward(self, x):
        N, _, _, _ = x.shape
        x = x.float()
        output = self.conv1(x)
        output = self.Relu_1(output)

        output = self.maxpooling_1(output)
        # 卷积层2

        output = self.conv2(output)
        output = self.Relu_2(output)

        output = self.conv3(output)
        output = self.Relu_3(output)

        output = self.maxpooling_3(output)
        # 卷积层3

        # 第一个全连接层
        output = output.view(output.size(0), -1)
        output = self.fc_1(output)
        output = self.sigmoid_1(output)
        # 第二个全连接层
        output = self.fc_2(output)
        output = self.softmax(output, dim=1)

        return output


train_set = EmotionClassificationDataset()
train_loader = torch.utils.data.DataLoader(dataset=train_set,
                                           batch_size=batch_size,
                                           shuffle=True)  # 将数据打乱
std_model = StdModel(num_class=7)
print('    Total params: %.2fM' % (sum(p.numel() for p in std_model.parameters()) / 1000000.0))
if use_gpu:
    std_model.cuda()
for p in std_model.parameters():
    p.requires_grad = True
optimizer = torch.optim.SGD(std_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
epochs = 20
for i in range(epochs):
    optimizer.zero_grad()
    right, total = 0, 0
    for batch_index, data in enumerate(train_loader):
        x = data[0]
        y = data[1]
        if use_gpu:
            x = x.cuda()
            y = y.cuda()
        x_pred = std_model(x)
        loss = criterion(x_pred, y)
        t1, t2 = cal_acc(x_pred, y)
        right += t1
        total += t2
        print('EPOCH: {}/{}, LOSS: {}'.format(i, epochs, loss.item()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch_index % 20 == 0:
            print("ACC: {}".format(right / total))
            right = total = 0
