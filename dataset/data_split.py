import numpy as np
import os
import argparse
import random


def ModelNet40_data_split(data_path, division_ratio=np.array([0.8, 0.1, 0.1])):
    """
    split ModelNet40 data

    :param data_path: ModelNet40 root
    :param division_ratio: split ratio [training set，val set，test set]
    :return:
    """

    assert division_ratio.shape == (3,), "Wrong division_ratio shape"

    division_ratio_norm = division_ratio / np.sum(division_ratio)
    # 转换为累加概率
    division_ratio_norm[1] += division_ratio_norm[0]
    division_ratio_norm[2] += division_ratio_norm[1]
    # 读取全部数据
    dir_names = os.listdir(data_path)
    data_files = []
    data_files_labels = []
    label = 0
    label_names = []
    for i, label_name in enumerate(dir_names):
        one_cls_data_path = os.path.join(data_path, label_name)
        if os.path.splitext(one_cls_data_path)[1]:
            continue
        one_cls_data_files = os.listdir(one_cls_data_path)
        one_cls_data_files_labels = [label] * len(one_cls_data_files)
        data_files.extend(one_cls_data_files)
        data_files_labels.extend(one_cls_data_files_labels)
        label_names.append(label_name)
        label += 1

    # 创建记录文件
    train_set_list_file = 'train_list.txt'
    validation_set_list_file = 'validation_list.txt'
    test_set_list_file = 'test_list.txt'
    # 划分数据集
    f_train = open(train_set_list_file, 'w')
    f_val = open(validation_set_list_file, 'w')
    f_test = open(test_set_list_file, 'w')
    for i, (data_file, label) in enumerate(zip(data_files, data_files_labels)):
        probability = random.uniform(0, 1)
        label_name = label_names[label]
        pointcloud_file = os.path.join(label_name, data_file)
        if probability <= division_ratio_norm[0]:
            # 划分为训练集
            f_train.write(pointcloud_file + ' ' + str(label) + '\n')
            pass
        elif probability <= division_ratio_norm[1]:
            # 划分为验证集
            f_val.write(pointcloud_file + ' ' + str(label) + '\n')
            pass
        else:
            # 划分为测试集
            f_test.write(pointcloud_file + ' ' + str(label) + '\n')
            pass
    f_train.close()
    f_val.close()
    f_test.close()
    print('Dataset split completed !')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default='/home/poley/Work_Space/Dataset/modelnet40_normal_resampled',
                        help='path of ModelNet40 data')
    parser.add_argument('--split_ratio', type=list,
                        default=[0.8, 0.1, 0.1],
                        help='split ratio')
    opt = parser.parse_args()
    ModelNet40_data_split(opt.dataset_path, np.array(opt.split_ratio))
