import torch
import numpy as np
import random

def seed_torch(seed=0):
 
    random.seed(seed)
 
    np.random.seed(seed)
 
    torch.manual_seed(seed)
 
    torch.cuda.manual_seed(seed)
 
    torch.cuda.manual_seed_all(seed) 
 
    torch.backends.cudnn.benchmark = False
 
    torch.backends.cudnn.deterministic = True

seed_torch(123456)
print('seed:{}'.format(torch.random.seed()))

x = torch.rand(3) * 2 - 1
print('x: {}'.format(x))
weight = torch.rand(3,1) * 2 - 1
print('weight: {}'.format(weight))

x_scale = x.abs().max()/64
print('x_scale: {}'.format(x_scale))
x_z = -64
x_q = torch.clamp(torch.round((x/x_scale-x_z)), min=0, max=127)
print('x_q: {}'.format(x_q))

otp = torch.clamp(torch.randint(low=-127,high=127,size=x_q.size()).to(torch.float32), min=-127, max=127)
print('otp: {}'.format(otp))

x_q_encoded = x_q + otp
print('x_q_encoded: {}'.format(x_q_encoded))

weight_scale = weight.abs().max()/(2**7-1)
print('weight_scale: {}'.format(weight_scale))
weight_q = torch.round(weight/weight_scale)
print('weight_q: {}'.format(weight_q))

result_q = x_q_encoded @ weight_q
print("result_q: {}".format(result_q))

result_decode_key = (otp - x_z) @ weight_q
print('decode_key: {}'.format(result_decode_key))

result = (result_q - result_decode_key) * x_scale * weight_scale
print('recovered_result: {}'.format(result))

print('real_quantized_result: {}'.format((x_q @ weight_q + (x_z * torch.ones_like(x_q))@weight_q)*x_scale*weight_scale))

print('real_float_result: {}'.format(x@weight))
