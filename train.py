import torch
from torch.utils.data import DataLoader
from dataset.ModelNet_dataset import ModelNet40_dataset
from models.PointNet_module import feature_transform_reguliarzer
from models.PointNet import PointNet_cls
from models.PointNetpp import PointNetpp_cls, get_loss
from tensorboardX import SummaryWriter
from tqdm import tqdm
import argparse
import torch.optim as optim
import torch.nn as nn
import os

def correct_and_accuracy(label,net_output):
    N=label.shape[0]
    pred_label=torch.argmax(net_output,dim=1)
    correct_num=torch.sum(label==pred_label)
    return correct_num.cpu().numpy() , correct_num.cpu().numpy()/N
def train(opt):
    print(opt)
    torch.cuda.empty_cache()
    if not os.path.exists(opt.record_folder):
        p=os.getcwd()
        os.mkdir(os.path.join(p,opt.record_folder))
    if not os.path.exists(opt.model_save_path):
        p=os.getcwd()
        os.mkdir(os.path.join(p,opt.model_save_path))


    train_dataset=ModelNet40_dataset(opt.dataset_path, 'dataset/train_list.txt', num_points=opt.num_points)
    train_dataloader=DataLoader(train_dataset,batch_size=opt.batchSize,shuffle=True,num_workers=opt.workers)

    val_dataset=ModelNet40_dataset(opt.dataset_path, 'dataset/validation_list.txt', num_points=opt.num_points)
    val_dataloader=DataLoader(val_dataset,batch_size=opt.batchSize,num_workers=opt.workers)

    test_dataset=ModelNet40_dataset(opt.dataset_path, 'dataset/test_list.txt', num_points=opt.num_points)
    test_dataloader=DataLoader(test_dataset,batch_size=opt.batchSize,num_workers=opt.workers)

    if opt.model=='pointnet++':
        net = PointNetpp_cls(num_class=40, normal_channel=False)
        save_net_file='PointNet++.pkl'
    elif opt.model=='pointnet':
        net=PointNet_cls(cls=40,normal_channel=False,transform=opt.feature_transform)
        save_net_file = 'PointNet.pkl'
    else:
        raise TypeError
    optimizer=optim.Adam(net.parameters(),lr=0.003)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    cls_loss_fun=get_loss()

    total_iter=-1
    record_path=os.path.join(opt.record_folder,str(opt.ind))
    w=SummaryWriter(log_dir= record_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    for epoch in range(opt.nepoch):
        net.train()
        print('Epoch %d begin...\n'%(epoch))
        print('Learning rate : %f\n'%(optimizer.param_groups[0]['lr']) )
        epoch_loss=0
        epoch_correct_num=0
        for iter_n,(data,label) in tqdm(enumerate(train_dataloader),total=train_dataloader.__len__()):
            #train_code
            optimizer.zero_grad()
            data = data.transpose(2,1)
            data = data.to(device)
            label = label.to(device)
            pred,trans_mat=net(data)
            cls_loss=cls_loss_fun(pred,label)
            if opt.feature_transform:
                tran_loss=feature_transform_reguliarzer(trans_mat[0])+feature_transform_reguliarzer(trans_mat[1])
                loss=cls_loss+0.001*tran_loss
            else:
                loss=cls_loss
            loss.backward()
            optimizer.step()
            #记录数据
            total_iter+=1
            iter_loss=loss.detach().cpu().numpy()
            w.add_scalar('train/iter_loss',iter_loss,total_iter)
            iter_correct_num,iter_accuracy=correct_and_accuracy(label,pred)
            epoch_correct_num+=iter_correct_num
            w.add_scalar('train/iter_accuracy', iter_accuracy, total_iter)
            epoch_loss+=iter_loss
            # if total_iter%opt.show_status_interval==0:
            #     print('current epoch: %d \n total iter : %d \n current loss : %f \n' %(epoch,total_iter,loss.detach().cpu().numpy()))
        epoch_loss=epoch_loss/iter_n
        print('--------------------------train result---------------------------')
        print('current epoch: %d \n train_loss : %f \n epoch acc : %f \n' % (
            epoch, epoch_loss, epoch_correct_num/train_dataset.__len__()))
        w.add_scalar('train/epoch_loss', epoch_loss, epoch)
        w.add_scalar('train/epoch_accuracy', epoch_correct_num/train_dataset.__len__(), epoch)
        scheduler.step()
        #保存网络
        torch.save(net, os.path.join(opt.model_save_path,save_net_file))
        val_epoch_correct_num=0
        val_epoch_loss=0
        net.eval()
        for iter_n, (data, label) in tqdm(enumerate(val_dataloader), total=val_dataloader.__len__()):
            #val_code
            data = data.transpose(2, 1)
            data = data.to(device)
            label = label.to(device)
            pred, trans_mat = net(data)
            cls_loss = cls_loss_fun(pred, label)
            if opt.feature_transform:
                tran_loss = feature_transform_reguliarzer(trans_mat[0]) + feature_transform_reguliarzer(trans_mat[1])
                loss = cls_loss + 0.01*tran_loss
            else:
                loss = cls_loss
            val_epoch_loss+=loss.detach().cpu().numpy()
            val_iter_correct_num, val_iter_accuracy = correct_and_accuracy(label, pred)
            val_epoch_correct_num+=val_iter_correct_num

        val_epoch_loss = val_epoch_loss / iter_n

        w.add_scalar('train/val_epoch_loss', val_epoch_loss, epoch)
        w.add_scalar('train/val_epoch_accuracy', val_epoch_correct_num / val_dataset.__len__(), epoch)
        print('--------------------------val result---------------------------')
        print('current epoch: %d \n val_loss : %f \n current acc : %f \n' % (
        epoch,val_epoch_loss,val_epoch_correct_num / val_dataset.__len__()))
    w.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batchSize', type=int, default=20, help='input batch size')
    parser.add_argument(
        '--num_points', type=int, default=3000, help='input batch size')
    parser.add_argument(
        '--workers', type=int, help='number of data loading workers', default=6)
    parser.add_argument(
        '--nepoch', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--dataset_path', type=str, default='/home/poley/Work_Space/Dataset/modelnet40_normal_resampled',
                        help='path of ModelNet40 data')
    parser.add_argument('--record_folder', type=str, default='./records', help='output folder of train record')
    parser.add_argument('--model_save_path', type=str, default='./save_models', help='model save path')
    parser.add_argument('--feature_transform', action='store_true', help="do not use feature transform")
    parser.add_argument('--show_status_interval', type=int, default='30',
                        help="interval of showing status during training ")
    parser.add_argument('--model', type=str, default='pointnet++',
                        help="choose the model")
    parser.add_argument('--ind', type=int, default=0,
                        help="index of running for record")
    opt = parser.parse_args()
    train(opt)