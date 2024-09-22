from torchvision import transforms
import torch
import os
import tqdm

import sys 
sys.path.append("../")
from utils import get_dataloader, get_float_models, get_optimizer
from config import get_options

opts = get_options()
opts.epoches = 40
# dataset
opts.dataset = 'cifar10'

# model
opts.model = 'vgg16'

# optim
opts.optimizer = 'sgd'
opts.lr =0.01
opts.weight_decay = 5e-3

test_transform, model = get_float_models(opts)

if opts.dataset =='cifar10':
    train_transform = transforms.Compose(
    [transforms.Pad(4),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
     transforms.RandomHorizontalFlip(),
     transforms.RandomGrayscale(),
     transforms.RandomCrop(32, padding=4),
])

elif opts.dataset == 'mnist':
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.406], [0.225])
    ])
    
else:
    exit('error model')
    
train_loader, _ = get_dataloader(opts=opts, transform=train_transform)
_, test_loader = get_dataloader(opts=opts, transform=test_transform)

optimizer = get_optimizer(opts)(model.parameters(), lr=opts.lr, weight_decay=opts.weight_decay)

loss_fn = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.4, last_epoch=-1)

total = 0
accuracy_rate = []
best_acc = 0

def test():
    model.eval()
    correct = 0  # 预测正确的图片数
    total = 0  # 总共的图片数
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(opts.device)
            outputs = model(images).to(opts.device)
            outputs = outputs.cpu()
            outputarr = outputs.numpy()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
    accuracy = 100 * correct / total
    accuracy_rate.append(accuracy)
    print(f'acc:{accuracy}%'.format(accuracy))
    return accuracy


for epoch in tqdm.tqdm(range(opts.epoches)):
    model.train()
    model.to(opts.device)
    running_loss = 0.0
    total_correct = 0
    total_trainset = 0
    for i, (data, labels) in enumerate(train_loader, 0):
        data = data.to(opts.device)
        outputs = model(data).to(opts.device)
        labels = labels.to(opts.device)
        loss = loss_fn(outputs, labels).to(opts.device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = outputs.max(1)
        correct = (pred == labels).sum().item()
        total_correct += correct
        total_trainset += data.shape[0]
        if i % 1000 == 0 and i > 0:
            print(f"正在进行第{i}次训练, running_loss={running_loss}".format(i, running_loss))
            running_loss = 0.0
    acc = test()
    if acc > best_acc:
        if not os.path.exists('../check_points/{}_{}'.format(opts.dataset,opts.model)):
            os.makedirs('../check_points/{}_{}'.format(opts.dataset,opts.model))
        torch.save(model.state_dict(), '../check_points/{}_{}/best.pth'.format(opts.dataset,opts.model))
    scheduler.step()