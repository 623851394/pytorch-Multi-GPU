import os
import math
import tempfile
import argparse
import torch
import torch.optim as optim
import torch.distributed as dist
from torchvision.datasets import mnist
from tqdm import tqdm
import sys

import torchvision.transforms as transforms
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import logging


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        # print("NOT using distributed mode")
        raise EnvironmentError("NOT using distributed mode")
        # args.distributed=False
        # return
    # print(args)
    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dis_backend = 'nccl'
    dist.init_process_group(
        backend=args.dis_backend,
        init_method=args.dis_url,
        world_size=args.world_size,
        rank=args.rank
    )
    dist.barrier()  # 等待所有gpu的运行到此处


def cleanup():
    dist.destroy_process_group()


class CNNNet(torch.nn.Module):

    def __init__(self, in_channel, out_channel_one, out_channel_two, out_channel_three, fc_1, fc_2, fc_out):
        super(CNNNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel_one, kernel_size=5, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2,padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=out_channel_one, out_channels=out_channel_two, kernel_size=5, stride=1,padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2,padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=out_channel_two,out_channels=out_channel_three, kernel_size=5, stride=1,padding=1)
        # 最后输出形状应该是 7*7*32 == 512
        # full link

        self.fc1 = torch.nn.Linear(5*5*32, fc_1)
        self.fc2 = torch.nn.Linear(fc_1, fc_2)
        self.output = torch.nn.Linear(fc_2, fc_out)

    def forward(self, x):
        x = self.pool1(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool2(torch.nn.functional.relu(self.conv2(x)))
        x = torch.nn.functional.relu(self.conv3(x))

        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.softmax(self.output(x), dim=1)

        return x

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()



def reduce_value(value, average=True):
    world_size = get_world_size()
    if world_size < 2:
        return value
    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value /= world_size
        return value

def is_main_process():
    return get_rank() == 0


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()


    if is_main_process() !=0:
        # print("rank is ", get_rank(), get_rank()==0, is_main_process() !=0)
        data_loader = tqdm(data_loader, file=sys.stdout)


    for step, data in enumerate(data_loader):
        images, labels = data
        images /= 255.0
        images = images.view(images.size(0),1,28,28)
        pred = model(images.to(device))

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step+1)
        if is_main_process():
            data_loader.desc = "[epoch {}] mean loss {} rank {}".format(epoch, round(mean_loss.item(), 3), get_rank())
        if not torch.isfinite(loss):
            print("WARNING : non-finite loss, end training", loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)
    return mean_loss.item()

def evaluate(model, data_loader, device):
    model.eval()

    sum_num = torch.zeros(1).to(device)
    e_acc  = 0
    if is_main_process() != 0:
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        images /= 255.0
        images = images.view(images.size(0),1,28,28)
        pred = model(images.to(device))


        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    sum_num = reduce_value(sum_num, average=False)

    return sum_num.item()

def main(args):
    
    
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training")
    init_distributed_mode(args)

    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size
    args.lr *= args.world_size
    checkpoint_path = ""
    if rank == 0:
        # file_wite = open("res.txt", 'w')
        # print("文件信息: ", file_wite)

        if os.path.exists("./weight") is False:
            os.makedirs("./weight")
        print(args)

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = mnist.MNIST('/data', train=True, transform=transform, download=False)
    test_dataset = mnist.MNIST("/data", train=False, transform=transform, download=False)

    train_sample = torch.utils.data.distributed.DistributedSampler(train_dataset)
    test_sample = torch.utils.data.distributed.DistributedSampler(test_dataset)

    train_batch_sampler = torch.utils.data.BatchSampler(
        train_sample, batch_size, drop_last=True
    )
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,batch_sampler=train_batch_sampler,pin_memory=True,num_workers=nw,
    )
    test_lodaer = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sample, pin_memory=True,num_workers=nw,
    )

    model = CNNNet(1, 16, 32, 32, 128, 64, 10)
    model = model.to(device)
    checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weight.pt")
    if rank==0:
        torch.save(model.state_dict(), checkpoint_path)
    dist.barrier()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.device])
    # pg = [p for p in model.parameters() if ]
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_sample.set_epoch(epoch)

        mean_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch
        )

        sum_num = evaluate(
            model, test_lodaer, device
        )
        acc = sum_num / test_sample.total_size

        if rank == 0:
            print("[epoch {}] accuracy {}".format(epoch, acc))
            # logging.info("[epoch {}] accuracy {}".format(epoch, acc))
            # print("开始写入文件")
            # with open("123.txt", 'w+') as file_wite:
            #     file_wite.write("[epoch {}] accuracy {}".format(epoch, acc))
            # print("当前目录", os.getcwd())
            torch.save(model.module.state_dict(), "./weight/weights.pth")


                # file_wite.flush()
            # if os.path.exists("res.txt"):
            #     print("存在文件")
            # else:
            #     print("不存在该文件")
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

        # if os.path.exists("123.txt"):
        #     print("存在文件")
        #     with open("123.txt", 'r') as f:
        #         print(f.readlines())
        # else:
        #     print("不存在该文件")

    cleanup()
    # if rank == 0:
        # file_wite.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--world_size', type=int, default=2)
    parser.add_argument('--dis_url', type=str, default='env://')
    args = parser.parse_args()

    main(args)
    