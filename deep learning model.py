import numpy as np
import pandas as pd
import torch
import torchio as tio
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import TensorDataset,DataLoader
import os
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
from torchvision import models as m
from sklearn.model_selection import train_test_split


directory = r"C:\Users\admin\Desktop\pancreas\all_data_pad_50_50_50.npy"
X = np.load(directory)
Y = pd.read_csv(r"C:\Users\admin\Desktop\pancreas\clinical_catergory.csv",index_col=0)
X = np.expand_dims(X, axis=1)
X = X.transpose(0,1,4,2,3)


#划分训练集和测试集
x_train,x_test,y_train,y_test = train_test_split(X,Y.event,test_size=0.3,random_state=42,stratify=Y.event)
x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train.values).long()
x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test.values).long()


#数据增强
training_transform = tio.Compose([

    tio.RandomMotion(p=0.2),
    tio.RandomBlur(p=0.2),
    tio.RandomNoise(p=0.2),
    tio.RandomFlip(p=0.2),
    tio.OneOf({
         tio.RandomAffine(): 0.8,
         tio.RandomElasticDeformation(): 0.2})

])

#定义dataset
class Train_dataset(data.Dataset):
    def __init__(self, img, label, transform):
        self.img = img
        self.training_transform = transform
        self.label = label

    def __getitem__(self, index):
        img = self.img[index]
        trans_img = self.training_transform(img)
        label = self.label[index]
        return trans_img, label

    def __len__(self):
        return len(self.img)


class Test_dataset(data.Dataset):
    def __init__(self, img, label):
        self.img = img
        self.label = label

    def __getitem__(self, index):
        img = self.img[index]
        label = self.label[index]
        return img, label

    def __len__(self):
        return len(self.img)


train_dataset = Train_dataset(x_train,y_train,training_transform)
test_dataset = Test_dataset(x_test,y_test)
train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=32)




#构建基于3d-resnet-18的网络架构
from resnet import generate_model
resnet18_ = generate_model(18)


class MyResNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 原resnet18架构第一层卷积核尺寸较大，步长较大，用于快速消减figure size，然对于我这个小的图片
        # 删除了原先紧接于stem layer的maxpooling层，并设置卷积层步长为2消减特征图尺寸，后紧接两个layer大层
        self.block1 = nn.Sequential(nn.Conv3d(1, 64, kernel_size=3, stride=2, padding=1)
                                    , resnet18_.bn1
                                    , resnet18_.relu)  # 删除池化层
        self.block2 = resnet18_.layer2
        self.block3 = resnet18_.layer3
        # 自适应平均池化+线形层都与残差网络一致
        self.avgpool = resnet18_.avgpool
        # 线形层的输出自己来写
        self.fc1 = nn.Linear(256, 2, bias=True)
        # self.relu = nn.ReLU(inplace=True)
        # self.dropout = nn.Dropout(0.2)
        # self.fc2 = nn.Linear(128,2,bias=True)

    def forward(self, x):
        x = self.block1(x)
        x = self.avgpool(self.block3(self.block2(x)))
        x = x.view(x.shape[0], 256)
        # x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc1(x)
        return x




device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = MyResNet().to(device)
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(),lr=0.0001)

epochs=100
for epoch in range(epochs):
    train_loss = 0
    train_correct = 0

    train_epoch_loss = []
    train_acc = []

    test_loss = 0
    test_correct = 0

    test_epoch_loss = []
    test_acc = []

    y_all = []
    outputs_all = []

    y_test_all = []
    outputs_test_all = []

    net.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        pred = net(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        train_loss += loss.item()
        y_all.extend(y.tolist())
        outputs_all.extend(torch.max(pred, dim=1)[0].tolist())
        with torch.no_grad():
            train_correct += (pred.argmax(1) == y).sum().item()
    auc_train = roc_auc_score(y_all, outputs_all)

    pth = "Epoch{}.pth"
    torch.save(net.state_dict(), os.path.join(save_path, pth).format(epoch))
    train_epoch_loss.append(train_loss / len(train_loader.dataset))
    train_acc.append(train_correct / len(train_loader.dataset))

    print("Epoch{}:[train loss:{},train_auc:{},train_acc:{}]".format(epoch, train_loss / len(train_loader.dataset),
                                                                     auc_train, train_acc))
    net.eval()
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = net(x)
        loss = loss_fn(pred, y)
        test_loss += loss.item()
        y_test_all.extend(y.tolist())
        outputs_test_all.extend(torch.max(pred, dim=1)[0].tolist())
        with torch.no_grad():
            test_correct += (pred.argmax(1) == y).sum().item()
    auc_test = roc_auc_score(y_test_all, outputs_test_all)

    test_epoch_loss.append(test_loss / len(test_loader.dataset))
    test_acc.append(test_correct / len(test_loader.dataset))
    print("Epoch{}:test loss:{},,auc_test:{}test_acc:{}".format(epoch, test_loss / len(test_loader.dataset), auc_test,
                                                                test_acc))



























