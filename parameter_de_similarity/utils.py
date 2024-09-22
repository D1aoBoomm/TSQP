import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.quantization import QConfig
from torch.quantization.observer import (
    MovingAverageMinMaxObserver,
    HistogramObserver,
    default_per_channel_weight_observer,
    MinMaxObserver,
    PerChannelMinMaxObserver,
    default_histogram_observer,
    MovingAveragePerChannelMinMaxObserver,
)
import copy
from torch.quantization import quantize_fx, FakeQuantize
from torch.quantization.quantize_fx import prepare_qat_fx, convert_fx, prepare_fx
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


def get_dataloader(opts, transform=None):
    dataset = opts.dataset
    test_data_root = opts.test_dataset_path
    train_data_root = opts.train_dataset_path

    if opts.shuffle == "True":
        shuffle = True
    else:
        shuffle = False

    if transform == None:
        transform = torchvision.transforms.ToTensor()
    else:
        transform = transform

    if dataset == "imagenet":

        val_dataset = torchvision.datasets.ImageFolder(
            root=test_data_root, transform=transform
        )
        val_dataset_loader = DataLoader(
            val_dataset,
            batch_size=opts.batch_size,
            shuffle=shuffle,
            num_workers=opts.num_workers,
        )

        train_dataset = torchvision.datasets.ImageFolder(
            root=train_data_root, transform=transform
        )
        train_dataset_loader = DataLoader(
            train_dataset,
            batch_size=opts.batch_size,
            shuffle=True,
            num_workers=opts.num_workers,
        )

        return train_dataset_loader, val_dataset_loader

    elif dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root="~/data/", train=True, download=True, transform=transform
        )
        train_dataset_loader = torch.utils.data.DataLoader(trainset, batch_size=24, shuffle=True)

        testset = torchvision.datasets.CIFAR10(
            root="~/data/", train=False, download=True, transform=transform
        )
        val_dataset_loader = torch.utils.data.DataLoader(testset, batch_size=24, shuffle=False)
        
        return train_dataset_loader, val_dataset_loader

    elif dataset == "mnist":
        from torchvision.datasets import mnist

        # transform = transforms.Compose([  # 已给定的概率随即水平翻转给定的PIL图像
        #     transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
        #     transforms.Normalize([0.406], [0.225])  # 用平均值和标准偏差归一化张量图像
        # ])

        val_dataset = mnist.MNIST(root=test_data_root, train=False, transform=transform)
        val_dataset_loader = DataLoader(
            val_dataset,
            batch_size=opts.batch_size,
            shuffle=shuffle,
            num_workers=opts.num_workers,
        )

        train_dataset = mnist.MNIST(
            root=train_data_root, train=True, transform=transform
        )
        train_dataset_loader = DataLoader(
            train_dataset,
            batch_size=opts.batch_size,
            shuffle=shuffle,
            num_workers=opts.num_workers,
        )

        return train_dataset_loader, val_dataset_loader

    else:
        exit("not completed dataset")


# return model和transform
def get_float_models(opts):

    model = opts.model

    # resnet18
    if model == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.DEFAULT
        transform = weights.transforms()
        model = resnet18(weights=weights)

        return transform, model

    # resnet50
    elif model == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights

        weights = ResNet50_Weights.DEFAULT
        transform = weights.transforms()
        model = resnet50(weights=weights).eval()

        return transform, model

    # vgg19
    elif model == "vgg19":
        from torchvision.models import vgg19, VGG19_Weights

        weights = VGG19_Weights.DEFAULT
        transform = weights.transforms()
        model = vgg19(weights=weights)

        return transform, model

    # mobilenet v3 small
    elif model == "mobilenetv3":
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

        weights = MobileNet_V3_Small_Weights.DEFAULT
        transform = weights.transforms()
        model = mobilenet_v3_small(weights=weights).eval()

        return transform, model

    # mobilenetv2
    elif model == "mobilenetv2":
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

        weights = MobileNet_V2_Weights.DEFAULT
        transform = weights.transforms()
        model = mobilenet_v2(weights=weights).eval()

        return transform, model

    elif model == "lenet5":
        from models.lenet5 import LeNet

        model = LeNet().eval()
        transform = transforms.Compose(
            [  # 已给定的概率随即水平翻转给定的PIL图像
                transforms.ToTensor(),  # 将图片转换为Tensor,归一化至[0,1]
                transforms.Normalize(
                    [0.406], [0.225]
                ),  # 用平均值和标准偏差归一化张量图像
            ]
        )

        return transform, model

    elif model == "vgg16":
        from models.vgg16 import VGG16
        
        vgg = [96, 96, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        model = VGG16(vgg).eval()
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        
        return transform, model

    else:
        exit("not complete model")


# return model和transform
def get_quantized_models(opts):

    model = opts.model
    qconfig_dict = {"": get_qconfig(opts.reduce_range)}

    # resnet18
    if model == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.DEFAULT
        transform = weights.transforms()
        model = resnet18(weights=weights)

    # resnet50
    elif model == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights

        weights = ResNet50_Weights.DEFAULT
        transform = weights.transforms()
        model = resnet50(weights=weights).eval()

    # vgg19
    elif model == "vgg19":
        from torchvision.models import vgg19, VGG19_Weights

        weights = VGG19_Weights.DEFAULT
        transform = weights.transforms()
        model = vgg19(weights=weights).eval()

    # mobilenet v3 small
    elif model == "mobilenetv3":
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

        weights = MobileNet_V3_Small_Weights.DEFAULT
        transform = weights.transforms()
        model = mobilenet_v3_small(weights=weights).eval()

    # mobilenetv2
    elif model == "mobilenetv2":
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

        weights = MobileNet_V2_Weights.DEFAULT
        transform = weights.transforms()
        model = mobilenet_v2(weights=weights).eval()
        replace_relu6_with_relu(model)
        model.eval()

    else:
        exit("not complete model")

    model_float32 = copy.deepcopy(model)
    example_inputs = (torch.randn(1, 3, 224, 224),)
    model_fp32_prepared = prepare_fx(
        model_float32, qconfig_dict, example_inputs=example_inputs
    )
    # print(model_fp32_prepared)

    return transform, model_fp32_prepared


# return model和transform
def get_fake_quantized_models(opts):

    model = opts.model

    # resnet18
    if model == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.DEFAULT
        transform = weights.transforms()
        model = resnet18(weights=weights)
        # model = torch.nn.Sequential(torch.ao.quantization.QuantStub(), model, torch.ao.quantization.DeQuantStub())

        qconfig_dict = {"": get_fakeq_qconfig(opts.reduce_range)}
        example_inputs = (torch.randn(1, 3, 224, 224),)

        model_float32 = copy.deepcopy(model).train()
        model_fp32_prepared = prepare_qat_fx(
            model_float32, qconfig_dict, example_inputs=example_inputs
        )

        return transform, model_fp32_prepared

    # resnet50
    elif model == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights

        weights = ResNet50_Weights.DEFAULT
        transform = weights.transforms()
        model = resnet50(weights=weights).eval()
        # model = torch.nn.Sequential(torch.ao.quantization.QuantStub(), model, torch.ao.quantization.DeQuantStub())

        qconfig_dict = {"": get_fakeq_qconfig(opts.reduce_range)}
        example_inputs = (torch.randn(1, 3, 224, 224),)

        model_float32 = copy.deepcopy(model).train()
        model_fp32_prepared = prepare_qat_fx(
            model_float32, qconfig_dict, example_inputs=example_inputs
        )

        return transform, model_fp32_prepared

    # vgg19
    elif model == "vgg19":
        from torchvision.models import vgg19, VGG19_Weights

        weights = VGG19_Weights.DEFAULT
        transform = weights.transforms()
        model = vgg19(weights=weights)
        # model = torch.nn.Sequential(torch.ao.quantization.QuantStub(), model, torch.ao.quantization.DeQuantStub())

        qconfig_dict = {"": get_fakeq_qconfig(opts.reduce_range)}
        example_inputs = (torch.randn(1, 3, 224, 224),)

        model_float32 = copy.deepcopy(model).train()
        model_fp32_prepared = prepare_qat_fx(
            model_float32, qconfig_dict, example_inputs=example_inputs
        )

        return transform, model_fp32_prepared

    # mobilenet v3 small
    elif model == "mobilenetv3":
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

        weights = MobileNet_V3_Small_Weights.DEFAULT
        transform = weights.transforms()
        model = mobilenet_v3_small(weights=weights).eval()
        # model = torch.nn.Sequential(torch.ao.quantization.QuantStub(), model, torch.ao.quantization.DeQuantStub())

        qconfig_dict = {"": get_fakeq_qconfig(opts.reduce_range)}
        example_inputs = (torch.randn(1, 3, 224, 224),)

        model_float32 = copy.deepcopy(model).train()
        model_fp32_prepared = prepare_qat_fx(
            model_float32, qconfig_dict, example_inputs=example_inputs
        )

        return transform, model_fp32_prepared

    # mobilenetv2
    elif model == "mobilenetv2":
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

        weights = MobileNet_V2_Weights.DEFAULT
        transform = weights.transforms()
        model = mobilenet_v2(weights=weights).eval()
        # model = torch.nn.Sequential(torch.ao.quantization.QuantStub(), model, torch.ao.quantization.DeQuantStub())

        qconfig_dict = {"": get_fakeq_qconfig(opts.reduce_range)}
        example_inputs = (torch.randn(1, 3, 224, 224),)

        model_float32 = copy.deepcopy(model).train()
        model_fp32_prepared = prepare_qat_fx(
            model_float32, qconfig_dict, example_inputs=example_inputs
        )

        return transform, model_fp32_prepared

    else:
        exit("not complete model")


# return model和transform
def get_qat_models(opts):

    model = opts.model

    # resnet18
    if model == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.DEFAULT
        transform = weights.transforms()
        model = resnet18(weights=weights)
        # model = torch.nn.Sequential(torch.ao.quantization.QuantStub(), model, torch.ao.quantization.DeQuantStub())
        # from torch.ao.quantization import get_default_qat_qconfig_mapping
        qconfig_dict = {"": get_qat_qconfig(opts.reduce_range)}
        example_inputs = (torch.randn(1, 3, 224, 224),)

        model_float32 = copy.deepcopy(model).train()
        model_fp32_prepared = prepare_qat_fx(
            model_float32, qconfig_dict, example_inputs=example_inputs
        )

        return transform, model_fp32_prepared

    # resnet50
    elif model == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights

        weights = ResNet50_Weights.DEFAULT
        transform = weights.transforms()
        model = resnet50(weights=weights).eval()
        # model = torch.nn.Sequential(torch.ao.quantization.QuantStub(), model, torch.ao.quantization.DeQuantStub())

        qconfig_dict = {"": get_qat_qconfig(opts.reduce_range)}
        example_inputs = (torch.randn(1, 3, 224, 224),)

        model_float32 = copy.deepcopy(model).train()
        model_fp32_prepared = prepare_qat_fx(
            model_float32, qconfig_dict, example_inputs=example_inputs
        )

        return transform, model_fp32_prepared

    # vgg19
    elif model == "vgg19":
        from torchvision.models import vgg19, VGG19_Weights

        weights = VGG19_Weights.DEFAULT
        transform = weights.transforms()
        model = vgg19(weights=weights)
        # model = torch.nn.Sequential(torch.ao.quantization.QuantStub(), model, torch.ao.quantization.DeQuantStub())

        qconfig_dict = {"": get_qat_qconfig(opts.reduce_range)}
        example_inputs = (torch.randn(1, 3, 224, 224),)

        model_float32 = copy.deepcopy(model).train()
        model_fp32_prepared = prepare_qat_fx(
            model_float32, qconfig_dict, example_inputs=example_inputs
        )

        return transform, model_fp32_prepared

    # mobilenet v3 small
    elif model == "mobilenetv3":
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

        weights = MobileNet_V3_Small_Weights.DEFAULT
        transform = weights.transforms()
        model = mobilenet_v3_small(weights=weights).eval()
        # model = torch.nn.Sequential(torch.ao.quantization.QuantStub(), model, torch.ao.quantization.DeQuantStub())

        qconfig_dict = {"": get_qat_qconfig(opts.reduce_range)}
        example_inputs = (torch.randn(1, 3, 224, 224),)

        model_float32 = copy.deepcopy(model).train()
        model_fp32_prepared = prepare_qat_fx(
            model_float32, qconfig_dict, example_inputs=example_inputs
        )

        return transform, model_fp32_prepared

    # mobilenetv2
    elif model == "mobilenetv2":
        from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

        weights = MobileNet_V2_Weights.DEFAULT
        transform = weights.transforms()
        model = mobilenet_v2(weights=weights).eval()
        # model = torch.nn.Sequential(torch.ao.quantization.QuantStub(), model, torch.ao.quantization.DeQuantStub())

        qconfig_dict = {"": get_qat_qconfig(opts.reduce_range)}
        example_inputs = (torch.randn(1, 3, 224, 224),)

        model_float32 = copy.deepcopy(model).train()
        model_fp32_prepared = prepare_qat_fx(
            model_float32, qconfig_dict, example_inputs=example_inputs
        )

        return transform, model_fp32_prepared

    else:
        exit("not complete model")


# 获取activation reduce_range的qconfig进行量化
def get_qconfig(reduce_range):
    # reduce_range = opts.reduce_range
    qconfig = QConfig(
        activation=HistogramObserver.with_args(
            quant_min=0,
            quant_max=255 - reduce_range,
            reduce_range=True,
            dtype=torch.quint8,
        ),
        #   activation=MinMaxObserver.with_args(quant_min=0, quant_max=255-reduce_range, reduce_range=True),
        # activation=default_histogram_observer,
        weight=default_per_channel_weight_observer,
    )
    # print(qconfig)
    return qconfig


# fakequantize模拟数值计算，用于比较不同量化区间的准确率
def get_fakeq_qconfig(reduce_range):
    # reduce_range = opts.reduce_range
    # intB_act_fq=FakeQuantize.with_args(observer=MinMaxObserver, quant_min=0, quant_max=255-reduce_range,  dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False)
    intB_act_fq = FakeQuantize.with_args(
        observer=HistogramObserver,
        quant_min=0,
        quant_max=255 - reduce_range,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
    )
    # intB_act_fq=FakeQuantize.with_args(observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255-reduce_range,  dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False)
    intB_weight_fq = FakeQuantize.with_args(
        observer=PerChannelMinMaxObserver,
        quant_min=-(2**8) // 2,
        quant_max=(2**8) // 2 - 1,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        reduce_range=False,
    )

    qconfig = QConfig(activation=intB_act_fq, weight=intB_weight_fq)

    # qconfig = torch.quantization.QConfig(
    #         activation = torch.quantization.HistogramObserver.with_args(
    #             quant_min=0,
    #             quant_max=1,
    #             dtype=torch.quint8,
    #             qscheme=torch.per_tensor_affine,
    #             reduce_range=False
    #         ),
    #        weight= default_per_channel_weight_observer)

    # print(qconfig)
    return qconfig

def desimilarity(pub_model, pri_model):
    # we are working to include the transformers and cv models together
    # coming soon
    return 0

def get_qat_qconfig(reduce_range):
    # best performence
    qconfig = QConfig(
        activation=FakeQuantize.with_args(
            observer=HistogramObserver,
            quant_min=0,
            quant_max=255 - reduce_range,
            reduce_range=True,
        ),
        weight=FakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            reduce_range=False,
            ch_axis=0,
        ),
    )
    # best speed
    qconfig = QConfig(
        activation=FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=0,
            quant_max=255 - reduce_range,
            reduce_range=True,
        ),
        weight=FakeQuantize.with_args(
            observer=MovingAveragePerChannelMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            reduce_range=False,
            ch_axis=0,
        ),
    )

    return qconfig


def replace_relu6_with_relu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU6):
            setattr(module, name, nn.ReLU())  # 将 ReLU6 替换为 ReLU
        else:
            replace_relu6_with_relu(child)


def get_optimizer(opts):
    if opts.optimizer == "sgd":
        return torch.optim.SGD

    elif opts.optimizer == "adam":
        return torch.optim.Adam

    elif opts.optimizer == "rmsprop":
        return torch.optim.RMSprop

    else:
        exit("未实现的优化器")


def distribution_draw(name=None, param=None):
    if isinstance(param, torch.Tensor):
        param = param.data.cpu().numpy()

    param = np.ravel(param)
    hist_values, bin_edges = np.histogram(param, bins=20)
    plt.clf()
    plt.hist(param, bins=20, edgecolor="black",color='#4e62ab')
    plt.xlabel("Param")
    plt.ylabel("Frequency")
    plt.title("Histogram {}".format(name))
    # plt.ylim(0, 40000)
    plt.xlim(param.min(), param.max())
    plt.grid(True)
    plt.tight_layout()
    plt.gca().set_facecolor('#EAEAF2')
    plt.savefig("./distribution_figure/Histogram_{}.png".format(name))
