import torch
import torch.nn as nn
import numpy as np


class STN(nn.Module):
    def __init__(self, k):
        super(STN, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k=k
    def forward(self, x):
        """
        :param x: size[batch,channel(feature_dim),L(length of signal sequence)]
        """
        batchsize = x.size()[0]
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = self.relu4(self.bn4(self.fc1(x)))
        x = self.relu5(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.tensor(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k**2).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetEncoder(nn.Module):
    def __init__(self,channel,transform=False):

        """
        PointNet Encoder

        :param channel: input channel
        :param global_feature: True：return global_feature False：return cat(points_feature,global_feature)
        :param transform: if use STN
        """
        super(PointNetEncoder, self).__init__()
        if transform:
            self.STN=STN(k=channel)
            self.STN_feature=STN(k=64)
        self.conv1=nn.Conv1d(channel,64,1)
        self.conv2=nn.Conv1d(64,64,1)
        self.conv3=nn.Conv1d(64,64,1)
        self.conv4=nn.Conv1d(64,128,1)
        self.conv5=nn.Conv1d(128,1024,1)

        self.bn1=nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.relu5 = nn.ReLU()

        # self.max_pool=nn.MaxPool1d(kernel_size=1,stride=1)

        self.transform=transform

    def forward(self,x,global_feature=True):
        #完全按论文所描述的结构来
        N=x.shape[2]
        if self.transform:
            tran1=self.STN(x)
            x=x.transpose(2,1)
            x=torch.bmm(x,tran1)
            x=x.transpose(2,1)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        if self.transform:
            tran2=self.STN_feature(x)
            x=x.transpose(2,1)
            x=torch.bmm(x,tran2)
            x=x.transpose(2,1)
        if self.transform:
            tran_mat=(tran1,tran2)
        else:
            tran_mat=None
        point_wise_feature=x
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.relu5(self.bn5(self.conv5(x)))

        global_feature_vec=torch.max(x, 2, keepdim=True)[0]
        global_feature_vec=global_feature_vec.view(-1,1024)
        if global_feature:
            return global_feature_vec,tran_mat
        else:
            temp=global_feature_vec.unsqueeze(2).repeat(1,1,N)
            return torch.cat([point_wise_feature,temp],dim=1),tran_mat

def feature_transform_reguliarzer(trans):
    """

    :param trans: Rotation Matrix
    :return: Regularization
    """
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    #网络得到的旋转矩阵应该尽量正交，并以正交性作为损失
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss




