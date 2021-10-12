import os
import argparse

"""
1、安装相关依赖
2、将pretrained-model移动至指定位置
"""
os.system('cd /home/work/user-job-dir/code && pip install -r requirements.txt')
os.system('cd /home/work/user-job-dir/code && '
          'mkdir -p /home/work/.cache/torch/hub/checkpoints/ &&'
          'cp ./pretrained/resnet50-19c8e357.pth '
          '/home/work/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth')

import moxing as mox
import time
import torch
from torch import nn
import torchvision
from torchvision import transforms

import utils

"""
消除随机因素的影响
"""
torch.manual_seed(2021)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a classification algorithm')
    parser.add_argument('--data_url', type=str, required=True,
                        help='the dataset dir of dataset')
    parser.add_argument('--train_url', type=str, required=True,
                        help='the checkpoint dir obs')
    parser.add_argument('--init_method', type=str, required=False,
                        help='')
    parser.add_argument('--num_gpus', type=int, required=False, default=1,
                        help='')
    parser.add_argument('--last_path', type=str, required=False,
                        help='')
    parser.add_argument('--train-dir', type=str, required=True,
                        help='the dataset dir of training dataset')
    parser.add_argument('--validate-dir', type=str, required=False,
                        default=None,
                        help='the dataset dir of validation dataset')
    parser.add_argument('--ckpt-dir', type=str, required=True,
                        help='the checkpoint dir')
    parser.add_argument('--num-classes', type=int, required=True,
                        help='num-classes, do not include bg')
    parser.add_argument('--batch-size', type=int, required=True,
                        help='batch size')
    parser.add_argument('--num-epochs', type=int, required=True,
                        help='the number of epochs')
    args = parser.parse_args()
    return args


def prepare_data(data_url, training_dir, validate_dir):
    print(data_url)
    # copy training dataset from obs data_url
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)
        print('mkdir: {}'.format(training_dir))
    mox.file.copy(os.path.join(data_url, '2021hwjx-data-train.zip'),
                  os.path.join(training_dir, '2021hwjx-data-train.zip'))
    os.system('ls {}'.format(training_dir))
    os.system('cd {} && unzip -qq 2021hwjx-data-train.zip'.format(
        training_dir))

    # copy validate dataset from obs data_url if validate_dir is not None
    if validate_dir is not None:
        print('process validate_dir')
        if not os.path.exists(validate_dir):
            os.makedirs(validate_dir)
            print('mkdir: {}'.format(validate_dir))
        mox.file.copy(os.path.join(data_url, '2021hwjx-data-val.zip'),
                      os.path.join(validate_dir, '2021hwjx-data-val.zip'))
        os.system('ls {}'.format(validate_dir))
        os.system('cd {} && unzip -qq 2021hwjx-data-val.zip'.format(
            validate_dir))


def load_data(dir, is_train):
    print("Loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if is_train:
        dataset = torchvision.datasets.ImageFolder(
            dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        dataset = torchvision.datasets.ImageFolder(
            dir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        sampler = torch.utils.data.SequentialSampler(dataset)

    return dataset, sampler


def initial_model(class_num):
    # load an classification model pre-trained on Imagenet
    model = torchvision.models.__dict__['resnet50'](pretrained=True)
    # get number of input features for the classifier
    channel_in = model.fc.in_features
    model.fc = nn.Linear(channel_in, class_num)
    return model


def train_one_epoch(model, criterion, optimizer, data_loader, idx_to_class, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('img/s', utils.SmoothedValue(window_size=10, fmt='{value}'))

    header = 'Epoch: [{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        acc0, acc0_5 = utils.accuracy(output, target, idx_to_class)
        acc = 0.5 * acc0 + 0.5 * acc0_5
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc0'].update(acc0.item(), n=batch_size)
        metric_logger.meters['acc0_5'].update(acc0_5.item(), n=batch_size)
        metric_logger.meters['acc'].update(acc.item(), n=batch_size)
        metric_logger.meters['img/s'].update(batch_size / (time.time() - start_time))
    print(' *Train acc@0 {top0.global_avg:.3f} Train acc@0.5 {top5.global_avg:.3f} Train acc@ {top.global_avg:.3f}'
          .format(top0=metric_logger.acc0, top5=metric_logger.acc0_5, top=metric_logger.acc))


def evaluate(model, criterion, data_loader, idx_to_class, device, print_freq=100):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc0, acc0_5 = utils.accuracy(output, target, idx_to_class)
            acc = 0.5 * acc0 + 0.5 * acc0_5
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc0'].update(acc0.item(), n=batch_size)
            metric_logger.meters['acc0_5'].update(acc0_5.item(), n=batch_size)
            metric_logger.meters['acc'].update(acc.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print(' *Eval acc@0 {top0.global_avg:.3f} Eval acc@0.5 {top5.global_avg:.3f} Eval acc@ {top.global_avg:.3f}'
          .format(top0=metric_logger.acc0, top5=metric_logger.acc0_5, top=metric_logger.acc))
    return metric_logger.acc0.global_avg


def main(args):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if \
        torch.cuda.is_available() else torch.device('cpu')

    train_dir = os.path.join(args.train_dir, 'train')
    dataset, train_sampler = load_data(train_dir, is_train=True)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=4, pin_memory=True)
    # provide idx and class relationgships
    idx_to_class = {value: key for key, value in dataset.class_to_idx.items()}

    if args.validate_dir is not None:
        val_dir = os.path.join(args.validate_dir, 'val')
        dataset_test, test_sampler = load_data(val_dir, is_train=False)
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=args.batch_size,
            sampler=test_sampler, num_workers=4, pin_memory=True)

    # get the model using our helper function
    model = initial_model(args.num_classes)

    # move model to the right device
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    print("Start training")
    for epoch in range(args.num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, criterion, optimizer, data_loader, idx_to_class, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        if args.validate_dir is not None:
            evaluate(model, criterion, data_loader_test, idx_to_class, device=device)
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args}

        if args.ckpt_dir is not None:
            if not os.path.exists(args.ckpt_dir):
                os.makedirs(args.ckpt_dir)
                print('mkdir: {}'.format(args.ckpt_dir))
        local_ckpt_path = os.path.join(args.ckpt_dir,
                                       'model_{}.pth'.format(epoch))

        # write classes.txt
        local_classes_path = os.path.join(args.ckpt_dir, 'classes.txt')
        with open(local_classes_path, 'w') as f:
            for i in range(len(idx_to_class)):
                f.write(idx_to_class[i] + "\n")

        # save model every 5 steps and upload model to obs
        if epoch % 5 == 0:
            utils.save_on_master(checkpoint, local_ckpt_path)

            if args.train_url is not None:
                # obs://obs-2021hwjx-baseline/data
                obs_target_path = os.path.join('{}'.format(args.train_url),
                                           os.path.basename(local_ckpt_path))
                # OBS 路径每隔30s会将内容同步到线上OBS桶中
                os.system("cp {} {}".format(local_ckpt_path, obs_target_path))
                print('finish upload {}->{}'.format(local_ckpt_path,
                                                obs_target_path))

    # save the final epoch model and upload it to obs
    utils.save_on_master(checkpoint, local_ckpt_path)
    if args.train_url is not None:
        # obs://obs-2021hwjx-baseline/data
        obs_target_path = os.path.join('{}'.format(args.train_url),
                                       os.path.basename(local_ckpt_path))
        # OBS 路径每隔30s会将内容同步到线上OBS桶中
        os.system("cp {} {}".format(local_ckpt_path, obs_target_path))
        print('finish upload {}->{}'.format(local_ckpt_path,
                                            obs_target_path))

    if local_ckpt_path is not None and args.last_path is not None:
        # 将最后一个model pth文件上传至模型发布路径下
        obs_target_path = os.path.join('{}'.format(args.last_path),
                                       'model_best.pth')
        print("upload {} -> {}".format(os.path.basename(local_ckpt_path),
                                       obs_target_path))
        os.system("cp {} {}".format(local_ckpt_path, obs_target_path))

        # 把classes.txt传到发布路径下
        class_target_path = os.path.join('{}'.format(args.last_path),
                                       'classes.txt')
        print("upload {} -> {}".format(os.path.basename(local_classes_path),
                                       class_target_path))
        os.system("cp {} {}".format(local_classes_path, class_target_path))


if __name__ == "__main__":
    args = parse_args()
    prepare_data(
        args.data_url,
        args.train_dir,
        args.validate_dir
    )
    print(args.data_url)
    print(args.train_url)
    os.system('ls {}'.format(args.train_url))
    main(args)
