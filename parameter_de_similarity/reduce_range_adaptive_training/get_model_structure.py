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

print(quantized_model)