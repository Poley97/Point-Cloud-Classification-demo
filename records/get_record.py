from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse


def get_record_data(opt):
    # load data
    rd = event_accumulator.EventAccumulator(opt.path)
    rd.Reload()
    print(rd.scalars.Keys())

    for dir in rd.scalars.Keys():
        subdir, item_name = dir.split('/')
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        data = rd.scalars.Items(dir)
        data = np.array([(i.step, i.value) for i in data])
        np.savetxt(dir + '.txt', data)

        # 绘图
        plt.figure()
        if item_name.split('_')[0] == 'epoch':
            plt.xlabel('Epochs')
            plt.title('Training Set')
        elif item_name.split('_')[0] == 'val':
            plt.xlabel('Epochs')
            plt.title('Validation Set')
        elif item_name.split('_')[0] == 'iter':
            plt.xlabel('Iterations')
            plt.title('Training Set')
        if item_name.split('_')[-1] == 'accuracy':
            plt.plot(data[:, 0], data[:, 1], label='Accuracy')
            plt.ylabel('Accuracy')
        if item_name.split('_')[-1] == 'loss':
            plt.plot(data[:, 0], data[:, 1], label='Loss')
            plt.ylabel('Loss')
        plt.legend()
        plt.grid()
        plt.savefig((dir + ".png").replace("_", ""))
        # plt.show()
        # plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='.\\0\events.out.tfevents.1617761562.poley-ubuntu',
                        help="path of events file")
    opt = parser.parse_args()
    get_record_data(opt)
