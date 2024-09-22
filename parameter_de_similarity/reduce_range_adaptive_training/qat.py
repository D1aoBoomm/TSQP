import torch
import torchvision
from tqdm import tqdm
import time
import copy
import os

import sys 
sys.path.append("../..") 

from config import get_options
from utils import get_dataloader,get_qat_models, get_optimizer, desimilarity
from torch.quantization.quantize_fx import convert_fx

def evaluate_model(model, test_loader, device=torch.device("cuda"), criterion=None):
    t0 = time.time()
    model = copy.deepcopy(model).to(device)
    model.apply(torch.ao.quantization.disable_observer)
    model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    # model.to(device)
    running_loss = 0
    running_corrects = 0
    
    with torch.inference_mode():
        for i,(inputs, labels) in enumerate(tqdm(test_loader)):

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            # print(labels.shape)
            # print(preds.shape)

            if criterion is not None:
                loss = criterion(outputs, labels).item()
            else:
                loss = 0

            # statistics
            running_loss += loss * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            if opts.debug_test_iter > 0:
                if i > opts.debug_test_iter:
                    break       

        eval_loss = running_loss / len(test_loader.dataset)
        
        if opts.debug_test_iter>0:
            eval_accuracy = running_corrects / (opts.debug_test_iter * opts.batch_size)
        else:
            eval_accuracy = running_corrects / len(test_loader.dataset)
        
        t1 = time.time()
    print(f"eval loss: {eval_loss}, eval acc: {eval_accuracy}, cost: {t1 - t0}")
    return eval_loss, eval_accuracy

opts = get_options()
opts.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 不用cuda慢死人

transform, prepared_model = get_qat_models(opts)
prepared_model = prepared_model.to(opts.device)
train_data_loader, val_data_loader = get_dataloader(opts, transform)
optimizer = get_optimizer(opts)
optimizer = optimizer(prepared_model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_step_size, gamma=opts.lr_gamma)

# QAT training

# 存储测试的
loss_list = []
acc_list = []

# 存储训练的
tra_loss_list = []
tra_acc_list = []

criterion = torch.nn.CrossEntropyLoss()

prepared_model.apply(torch.ao.quantization.enable_observer)
prepared_model.apply(torch.ao.quantization.enable_fake_quant)
for epoch in range(opts.epoches):
    train_loss = 0
    train_acc = 0
    print('epoch:{}/{}'.format(epoch, opts.epoches))
    for i, (data, label) in enumerate(tqdm(train_data_loader)):
        data, label = data.to(opts.device), label.to(opts.device)
        optimizer.zero_grad()
        
        output = prepared_model(data)
        loss = criterion(output, label) # +  desimilarity()
        
        train_loss += loss.item()
                
        loss.backward()
        optimizer.step()
        
        #if(i>10000):
        #    break
       
    print('Train Loss: {:.6f}'.format(train_loss / (len(train_data_loader))))#输出训练时的loss和acc
    lr_scheduler.step()
        
    if epoch >3: #关闭observer
        prepared_model.apply(torch.ao.quantization.disable_observer)
    
    if epoch >4: #关闭bn统计
        prepared_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
    
    # 测试
    print('正在测试模型')
    evaluate_model(prepared_model, val_data_loader, torch.device("cpu"), criterion=None)
    
    if not os.path.exists('./models/{}'.format(opts.model)):
        os.makedirs('./models/{}'.format(opts.model))
    torch.save(prepared_model.state_dict(), './models/{}/{}_reduce_range_{}_epoch_{}.pth'.format(opts.model, opts.model, opts.reduce_range, epoch))
    
