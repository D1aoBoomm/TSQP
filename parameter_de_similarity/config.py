import argparse
import torch
import numpy as np

def parse_arguments():
    parser = argparse.ArgumentParser(description='训练参数咯')

    '''总体设置'''
    parser.add_argument("--random_seed", type=int, default=-1)
    parser.add_argument("--exp_num", type=int, default=15)

    '''数据集设置'''
    parser.add_argument("--train_dataset_path", type=str, default='~/data/ImageNet/train')
    parser.add_argument("--test_dataset_path", type=str, default='~/data/ImageNet/val')
    parser.add_argument('--dataset', type=str, default='imagenet') # imagenet
    parser.add_argument('--shuffle', type=str, default='True')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    
    '''量化设置'''
    parser.add_argument('--reduce_range', type=int, default=0) # reduce_range 的大小
    parser.add_argument('--cali_num', type=int, default=150) # 校准的数据数量
    parser.add_argument('--fake_quantization', type=bool, default=False) # 是否用伪量化模拟PTQ,VGG可以用，其他不行
    
    '''训练设置'''
    parser.add_argument('--optimizer', type=str, default='sgd') # sgd, adam
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--epoches', type=int, default=90)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr_step_size', type=float, default=30)
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--debug_test_iter', type=int, default=0)
    
    '''扰动的百分比'''
    parser.add_argument('--noise', type=float, default=0.0)

    '''模型存取设置'''
    parser.add_argument('--model', type=str, default='resnet18')
    # resnet18, vgg19, resnet50, mobilenetv3, mobilenetv2
    
    parser.add_argument('--load_path', type=str, default=None) # 如果不为空就从路径读取保存的模型信息

    args = parser.parse_args()
    return args

def get_options():
    opt = parse_arguments()

    # 确定是否采取随机数种子
    if opt.random_seed > 0:
        torch.manual_seed(opt.random_seed)
        print('Seed is set as:{}'.format(opt.random_seed))
    else:
        print('No Seed is set')
    
    # 能用显卡就用吧
    device = torch.device("cuda" if (torch.cuda.is_available()) and torch.cuda.device_count() > 0 else "cpu")
    opt.device = device
    # opt.device = 'cpu'

    # 忽略警告
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    print(opt)

    return opt
