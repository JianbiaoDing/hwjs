import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import MyDataset
import torch.nn as nn
import torch.optim as optim
import time
from model import get_model, LineNet, LineNet_cnn, LineNet_cnn1
import numpy as np


def CrossEntropyLoss_label_smooth(outputs, targets,
                                  num_classes=14, epsilon=0.1):
    N = targets.size(0)
    smoothed_labels = torch.full(size=(N, num_classes), fill_value=epsilon / (num_classes - 1))
    targets = targets.data.cpu()
    smoothed_labels.scatter_(dim=1, index=torch.unsqueeze(targets, dim=1), value=1 - epsilon)

    log_prob = nn.functional.log_softmax(outputs, dim=1).cpu()
    loss = - torch.sum(log_prob * smoothed_labels) / N
    return loss


def my_loss(outputs, targets, num_classes=14):
    batch_size = targets.size(0)
    class_num = num_classes
    one_hot = np.eye(class_num)  # 10*10的矩阵 对角线上是1

    one_hot_label = one_hot[targets.cpu().numpy().astype(np.int)]  #

    for i in range(batch_size):
        if targets[i] == 0:
            one_hot_label[i][0] = 0.9
            one_hot_label[i][1] = 0.1
        elif targets[i] == 13:
            one_hot_label[i][-1] = 0.9
            one_hot_label[i][-2] = 0.1
        else:
            k = targets[i]
            one_hot_label[i][k] = 0.8
            one_hot_label[i][k + 1] = 0.1
            one_hot_label[i][k - 1] = 0.1
    # import pdb
    # pdb.set_trace()
    one_hot_label = torch.tensor(one_hot_label)
    log_prob = nn.functional.log_softmax(outputs, dim=1).cpu()
    loss = - torch.sum(log_prob * one_hot_label) / batch_size
    return loss


def get_train_val_data(train_path, val_path, batch_size, dstSize):
    image_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.1),  # 水平翻转
            transforms.RandomVerticalFlip(p=0.1),  # 垂直翻转
            transforms.RandomRotation(90),
            transforms.RandomCrop(896)  # 随机裁剪896大小的图
            # transforms.RandomResizedCrop(112, scale=(0.8, 1.0)),  # 随机裁剪出不同大小、不同长宽比的图片，并缩放至112大小
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1)  # 改变图像的属性：亮度（brightness）、对比度（contrast）、饱和度（saturation）和色调（hue)
        ]),
        'val': transforms.Compose([transforms.CenterCrop(896)])
    }

    labels = ['6.5', '7.0', '7.5', '8.0', '8.5', '9.0', '9.5', '10.0', '10.5', '11.0', '11.5', '12.0', '12.5', '13.0']

    train_data = None
    valid_data = None
    if train_path:
        train_dataset = MyDataset(train_path, dstSize, transform=image_transforms['train'], labels=labels)
        train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if val_path:
        val_dataset = MyDataset(val_path, dstSize, transform=None, labels=labels)
        valid_data = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    return train_data, valid_data


def train_model(model, train_data, val_data, device, epochs=25, lr=0.01):
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # 每训练step_size个epoch，更新一次参数
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=2)

    best_acc = 0
    best_epoch = 0
    train_data_size = len(train_data.dataset.img_path)
    valid_data_size = len(val_data.dataset.img_path) if val_data else 0

    total_iter = train_data_size // train_data.batch_size

    for epoch in range(epochs):
        t1 = time.time()
        print("Epoch: {}/{}".format(epoch + 1, epochs))

        train_loss = 0.0
        train_acc = 0.0
        valid_loss = 0.0
        valid_acc = 0.0

        model.train()
        count = 0
        t1 = time.time()
        for i, (inputs, labels) in enumerate(train_data):
            inputs = inputs.to(device)

            # one_hot = torch.nn.functional.one_hot(labels, n_classes)
            # labels = labels.to(device, dtype=torch.float32)
            labels = labels.to(device)

            # 因为这里梯度是累加的，所以每次要清零
            optimizer.zero_grad()

            outputs = model(inputs)
            # loss = criterion(outputs, labels)
            loss = my_loss(outputs, labels)
            loss.backward()  # 计算好梯度
            optimizer.step()  # 更新参数

            train_loss += loss.item() * inputs.size(0)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)

            count += inputs.size(0)
            print("\r" + 'lr {}, iter {}/{}, loss {:.4f} , acc {:.4f}, train time {:.1f}m'.
                  format(optimizer.param_groups[0]['lr'], i, total_iter, train_loss / count, train_acc / count, (time.time() - t1) / 60), end='')

        avg_train_loss = train_loss / train_data_size
        avg_train_acc = train_acc / train_data_size

        if val_data:
            with torch.no_grad():
                model.eval()

                for j, (inputs, labels) in enumerate(val_data):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    valid_loss += loss.item() * inputs.size(0)

                    ret, predictions = torch.max(outputs.data, 1)
                    correct_counts = predictions.eq(labels.data.view_as(predictions))

                    acc = torch.mean(correct_counts.type(torch.FloatTensor))
                    valid_acc += acc.item() * inputs.size(0)

        avg_valid_loss = valid_loss / valid_data_size if valid_data_size > 0 else 0
        avg_valid_acc = valid_acc / valid_data_size if valid_data_size > 0 else 0

        if best_acc < avg_valid_acc:
            best_acc = avg_valid_acc
            best_epoch = epoch + 1

        t2 = time.time()
        print('\n' + "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, Validation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}m".format(
                epoch + 1, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100, (t2 - t1) / 60))

        # print("Best Accuracy for validation : {:.4f} at epoch {:03d}".format(best_acc, best_epoch))
        torch.save(model.state_dict(), 'checkpoints/e{}_lr_{:.6f}_train-acc_{:.4f}_val-acc_{:.4f}.pt'.format(epoch, optimizer.param_groups[0]['lr'], avg_train_acc, avg_valid_acc))

        # scheduler.step(avg_valid_acc)
        # scheduler.step(avg_train_acc)
        scheduler.step()


if __name__ == '__main__':
    # train_path = 'data/train'
    # val_path = 'data/val'

    train_path = 'data/src_sorted'
    val_path = None

    # pretrained_model_path = 'checkpoints/e10_lr_0.000500_train-acc_0.6260_val-acc_0.0000.pt'
    pretrained_model_path = None

    device = 'cuda' if torch.cuda.is_available() else "cpu"
    n_classes = len(os.listdir(train_path))  # 类别

    src_size = (1376, 1104)  # w, h
    dst_h = 512
    dst_w = int(src_size[0] / src_size[1] * dst_h)
    dst_w = dst_w - dst_w % 4  # 四的倍数
    dstSize = (dst_w, dst_h)
    dstSize = (368, 368)

    model = get_model(n_classes, pretrained=True)
    # model = LineNet(n_classes, dstSize)
    # model = LineNet_cnn1(n_classes, dstSize)
    if pretrained_model_path:
        model.load_state_dict(torch.load(pretrained_model_path))
    model.to(device)

    epochs = 100
    # lr = 0.001
    lr = 0.005
    batch_size = 20

    train_data, val_data = get_train_val_data(train_path, val_path, batch_size, dstSize)
    train_model(model, train_data, val_data, device, epochs=epochs, lr=lr)

