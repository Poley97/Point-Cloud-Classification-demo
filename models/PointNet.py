from models.PointNet_module import PointNetEncoder
import torch.nn as nn

class PointNet_cls(nn.Module):
    def __init__(self,cls=40,normal_channel=False,transform=False):
        super(PointNet_cls, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(channel=channel,transform=transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, cls)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.log_softmax=nn.LogSoftmax(dim=1)
    def forward(self, x):
        x, trans_mat = self.feat(x)
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = self.log_softmax(x)
        return x, trans_mat