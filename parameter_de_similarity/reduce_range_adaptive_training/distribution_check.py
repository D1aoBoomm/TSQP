import torch
import torchvision
from tqdm import tqdm
import time
import copy
import os

import sys 
sys.path.append("../..") 

from config import get_options
from utils import get_dataloader,get_qat_models, get_float_models, distribution_draw
from torch.quantization.quantize_fx import convert_fx

opts = get_options()
opts.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 不用cuda慢死人

transform, prepared_model = get_qat_models(opts) # 获取准备模型
transform, float_model = get_float_models(opts)

prepared_model.load_state_dict(torch.load(opts.load_path))
quantized_model = convert_fx(prepared_model)

for png in os.listdir('./distribution_figure'):
    os.remove(os.path.join('./distribution_figure', png))

# 看量化后模型参数分布
# print('quantized_model---------------------------------------')
quantized_params_dict = quantized_model.state_dict()
float_params_dict = float_model.state_dict()
print('name quantized_var float_var')
for param_name in quantized_params_dict.keys():
    if 'weight' in param_name:
        # print(quantized_params_dict[param_name])
        print(param_name, torch.var(quantized_params_dict[param_name].int_repr().to(torch.float32)), torch.var(float_params_dict[param_name]))
        distribution_draw(name=param_name+'_quantized', param=quantized_params_dict[param_name].int_repr().to(torch.float32))
        distribution_draw(name=param_name+'_float', param=float_params_dict[param_name])
# print('float_model---------------------------------------')
# for param_name in float_params_dict.keys():
#     if 'weight' in param_name and not 'bn' in param_name:
#         # print(quantized_params_dict[param_name])
#         print(param_name, torch.var(float_params_dict[param_name]))