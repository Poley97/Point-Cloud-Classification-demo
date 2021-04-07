import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
from scipy.spatial.transform import Rotation as R
import random


class ModelNet40_dataset(Dataset):
    def __init__(self, data_path, split_file, augmentation=True, normal_data=False, num_points=2000):
        """
        # read data base on the data set split results given by data_split.py

        :param data_path: ModelNet40 root
        :param augmentation: if augmentation
        :param normal_data: if contain normal
        """
        self.data_path = data_path  # 文件目录
        self.augmentation = augmentation  # 是否数据增强
        self.label_names = os.listdir(self.data_path)  # 读取数据文件列表
        self.normal_data = normal_data
        self.data_files = []
        self.data_files_labels = []
        self.num_points = num_points
        with open(split_file, "r") as f:
            for line in f.readlines():
                file_path, label = line.strip('\n').split(' ')
                self.data_files.append(file_path)
                self.data_files_labels.append(label)

    def __len__(self):  # 返回整个数据集的大小
        return len(self.data_files)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        data_file = self.data_files[index]  # 根据索引index获取路径
        pointcloud_file = os.path.join(self.data_path, data_file)
        data = np.loadtxt(pointcloud_file, delimiter=',', dtype=np.float32)
        label = np.array(self.data_files_labels[index], dtype=np.int64)
        N = data.shape[0]
        choice_index = np.random.choice(range(N), size=self.num_points)
        data = data[choice_index, :]
        if not self.normal_data:
            data = data[:, 0:3]
        if self.augmentation:
            data = self.augmentation_rotate(data)  # 对样本进行变换
        return data, label  # 返回该样本

    def augmentation_rotate(self, data):
        """

        :param data: origin pointcloud
        :return: augmented data
        """

        rotate_angle = random.uniform(-np.pi, np.pi) / np.pi
        r = R.from_euler('xzy', [0, 0, rotate_angle])
        r1 = r.as_matrix()
        aug_data = np.matmul(np.linalg.inv(r1), data.T).T
        return aug_data.astype(np.float32)


class ModelNet40_full_dataset(Dataset):
    def __init__(self, data_path, augmentation=True, normal_data=False):
        """
        # read all ModelNet40 data without split

        :param data_path: ModelNet40 root
        :param augmentation: if augmentation
        :param normal_data: if contain normal
        """
        self.data_path = data_path  # 文件目录
        self.augmentation = augmentation  # 是否数据增强
        self.label_names = os.listdir(self.data_path)  # 读取数据文件列表
        self.normal_data = normal_data
        self.data_files = []
        self.data_files_labels = []
        for i, label_name in enumerate(self.label_names):
            one_cls_data_path = os.path.join(self.data_path, label_name)
            if os.path.splitext(one_cls_data_path)[1]:
                continue
            one_cls_data_files = os.listdir(one_cls_data_path)
            one_cls_data_files_labels = [i] * len(one_cls_data_files)
            self.data_files.extend(one_cls_data_files)
            self.data_files_labels.extend(one_cls_data_files_labels)

    def __len__(self):  # 返回整个数据集的大小
        return len(self.data_files)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        data_file = self.data_files[index]  # 根据索引index获取路径
        label_name = self.label_names[self.data_files_labels[index]]
        pointcloud_file = os.path.join(self.data_path, label_name, data_file)
        data = np.loadtxt(pointcloud_file, delimiter=',', dtype=np.float32)
        label = np.array(self.data_files_labels[index], dtype=np.long)
        N = data.shape[0]
        choice_index = np.random.choice(range(N), size=self.num_points)
        data = data[choice_index, :]
        if not self.normal_data:
            data = data[:, 0:3]
        if self.augmentation:
            data = self.augmentation_rotate(data)  # 对样本进行变换
        return data, label  # 返回该样本

    def augmentation_rotate(self, data):
        """

        :param data: origin pointcloud
        :return: augmented data
        """

        rotate_angle = random.uniform(-10, 10) / np.pi
        r = R.from_euler('xyz', [0, 0, rotate_angle])
        r1 = r.as_matrix()
        aug_data = np.matmul(np.linalg.inv(r1), data.T).T
        return aug_data.astype(np.float32)
