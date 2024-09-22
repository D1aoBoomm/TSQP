import torch
import torchvision
from tqdm import tqdm


import sys 
sys.path.append("../..") 
from utils import get_dataloader, get_fake_quantized_models, get_fakeq_qconfig, get_quantized_models
from config import get_options
from torch.quantization.quantize_fx import prepare_qat_fx,convert_fx, prepare_fx

opts = get_options()
_, dataloader = get_dataloader(opts)

if opts.fake_quantization:
    transform, model_fp32_prepared = get_fake_quantized_models(opts) # 伪量化
    opts.device = 'cuda'
else:
    transform, model_fp32_prepared = get_quantized_models(opts)
    opts.device = 'cpu'
    
model_fp32_prepared = model_fp32_prepared.to(opts.device)
# print(model_fp32_prepared)

# 校准
if opts.fake_quantization:
    model_fp32_prepared.apply(torch.ao.quantization.disable_fake_quant) # 关闭伪量化
    model_fp32_prepared.apply(torch.ao.quantization.enable_observer)    # 启用观测器
# print(model_fp32_prepared)

with torch.inference_mode():
    cal_count = 0
    for data, label in tqdm(dataloader):
        data = data.to(opts.device)
        data = transform(data)
        model_fp32_prepared(data)
        cal_count += 1
        if cal_count > opts.cali_num:
            break

# 测试
if opts.fake_quantization:
    model_fp32_prepared.apply(torch.ao.quantization.enable_fake_quant) # 启动伪量化
    model_fp32_prepared.apply(torch.ao.quantization.disable_observer) # 避免观测数据再变化
# model_fp32_prepared.print_readable()

#不是伪量化就先转换一下
if not opts.fake_quantization:
    model_fp32_prepared = convert_fx(model_fp32_prepared)

correct_count = 0
total_count = 0

with torch.inference_mode():
    for data, label in tqdm(dataloader):
        data = transform(data)
        
        data = data.to(opts.device)
        label = label.to(opts.device)
        
        output = model_fp32_prepared(data)

        predict_label = torch.argmax(output)
        
        if predict_label == label:
            correct_count += 1
        # total_count += 1
        
        # if total_count > 1: #试试小批量准确率，看模型有无问题
        #     break

torch.save(model_fp32_prepared.state_dict() , './models/{}_reduce{}.pth'.format(opts.model,opts.reduce_range))

print('net:{} reduce_range:{} accuracy:{:.7f}'.format(opts.model, opts.reduce_range, correct_count/total_count))

f = open('./{}_result.txt'.format(opts.model),'a+')
f.write('net:{} reduce_range:{} accuracy:{:.7f}\n'.format(opts.model, opts.reduce_range, correct_count/total_count))