import os
from collections import OrderedDict
import argparse
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models, datasets
import model.resnet_cbam as resnet_cbam
from trainer.trainer import Trainer
from utils.logger import Logger
from torchnet.meter import ClassErrorMeter
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn


def main(args):

    # region -----------------------------------------记录训练日志-----------------------------------------

    if 0 == len(args.resume):
        logger = Logger('./logs/'+args.model+'.log')
    else:
        logger = Logger('./logs/'+args.model+'.log', True)

    logger.append(vars(args))

    # if args.display:
    #     writer = SummaryWriter()
    # else:
    #     writer = None
    writer = SummaryWriter()

    # endregion

    # region ----------------------------------------数据预处理配置----------------------------------------
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    # endregion

    # region -------------------------------------------数据加载-------------------------------------------

    train_datasets = datasets.CIFAR10(
        root= "data",
        train =True,
        download= True,
        transform = transforms.ToTensor()
    )
    val_datasets = datasets.CIFAR10(
        root= "data",
        train =False,
        download= True,
        transform = transforms.ToTensor()
    )



    # train_datasets = datasets.ImageFolder(os.path.join(args.data_root, 't256'), data_transforms['train'])
    # val_datasets   = datasets.ImageFolder(os.path.join(args.data_root, 'v256'), data_transforms['val'])
    train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_dataloaders   = torch.utils.data.DataLoader(val_datasets, batch_size=1024, shuffle=False, num_workers=4)
    # endregion

    # region --------------------------------------网络无关配置设置----------------------------------------
    # 记录日志
    if args.debug:
        x, y =next(iter(train_dataloaders))
        logger.append([x, y])

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    is_use_cuda = torch.cuda.is_available()
    cudnn.benchmark = True

    # 设置基础网络
    if  'resnet50' == args.model.split('_')[0]:
        my_model = models.resnet50(pretrained=False)
    elif 'resnet50-cbam' == args.model.split('_')[0]:
        my_model = resnet_cbam.resnet50_cbam(pretrained=False)
    elif 'resnet101' == args.model.split('_')[0]:
        my_model = models.resnet101(pretrained=False)
    else:
        raise ModuleNotFoundError

    # endregion

    # region --------------------------------------网络训练配置设置----------------------------------------
    # 损失函数设定
    loss_fn = [nn.CrossEntropyLoss()]
    # 优化器设置
    optimizer = optim.SGD(my_model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    # 学习率优化函数设置
    lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
    # 累计误差TOP5
    metric = [ClassErrorMeter([1,5], True)]
    # 迭代次数
    epoch  = int(args.epoch)
    # 传入训练器
    my_trainer = Trainer(my_model, args.model, loss_fn, optimizer, lr_schedule, 500, is_use_cuda, train_dataloaders,
                        val_dataloaders, metric, 0, epoch, args.debug, logger, writer)
    # 训练
    my_trainer.fit()
    logger.append('训练完毕')
    # endregion

if __name__ == '__main__':

    # region ------------------------------------------参数设定--------------------------------------------

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-r', '--resume', default='', type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--debug', action='store_true', dest='debug',
                        help='trainer debug flag')
    parser.add_argument('-g', '--gpu', default='0', type=str,
                        help='GPU ID Select')                    
    parser.add_argument('-d', '--data_root', default='./datasets',
                         type=str, help='data root')
    parser.add_argument('-t', '--train_file', default='./datasets/train.txt',
                         type=str, help='train file')
    parser.add_argument('-v', '--val_file', default='./datasets/val.txt',
                         type=str, help='validation file')
    parser.add_argument('-m', '--model', default='resnet101',
                         type=str, help='model type')
    parser.add_argument('--batch_size', default=32,
                         type=int, help='model train batch size')
    parser.add_argument('--display', action='store_true', dest='display',
                        help='Use TensorboardX to Display')
    parser.add_argument('--epoch', default='1',
                        help='epoch size')
    args = parser.parse_args()

    #endregion

    main(args)