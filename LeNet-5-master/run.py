from lenet import LeNet5
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# import visdom
import onnx
import sys
import time
from modelsize_estimate import modelsize
from gpu_mem_track import MemTracker

# viz = visdom.Visdom()

if len(sys.argv) == 2:
    t = time.localtime()
    if t.tm_hour >= 16:
        tm_mday = t.tm_mday + 1
    else:
        tm_mday = t.tm_mday
    layer_mem_log = "./log/LeNet_%d_%d_%d_%d_%s_layers.txt" % (
        t.tm_mon, tm_mday, (t.tm_hour + 8) % 24, t.tm_min, sys.argv[1])
    mem_log = "./log/LeNet_%d_%d_%d_%d_%s.txt" % (t.tm_mon, tm_mday, (t.tm_hour + 8) % 24, t.tm_min, sys.argv[1])

# 预处理数据集，resize大小，数据转化为tensor格式
data_train = MNIST('./data/mnist',
                   download=True,
                   transform=transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
data_test = MNIST('./data/mnist',
                  train=False,  # 区分训练集和测试集
                  download=True,
                  transform=transforms.Compose([
                      transforms.Resize((32, 32)),
                      transforms.ToTensor()]))
# 加载，num_workers为额外线程用来为主线程加载数据
# 训练集中60000张图，一个batch中放batch_size张图片，这里输入图形形状为（batch_size,1,32,32）
data_train_loader = DataLoader(data_train, batch_size=1024, shuffle=True, num_workers=8)
data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Track the amount of GPU memory usage
gpu_tracker = MemTracker()

net = LeNet5(layer_mem_log).to(device)
# Calculate the memory usage of a single model
inp = iter(data_train_loader).next()[0]
inp = inp.to(device)
print(inp.shape)
modelsize(net, inp)

# 损失函数（交叉熵）
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = optim.Adam(net.parameters(), lr=2e-3)


# cur_batch_win = None
# cur_batch_win_opts = {
#     'title': 'Epoch Loss Trace',
#     'xlabel': 'Batch Number',
#     'ylabel': 'Loss',
#     'width': 1200,
#     'height': 600,
# }


def train(epoch):
    with open(mem_log, "a+") as f:
        # global cur_batch_win
        net.train()  # train模式，启用 Batch Normalization 和 Dropout
        loss_list, batch_list = [], []
        for i, (images, labels) in enumerate(data_train_loader):
            print(images.dtype)
            print(images.is_cuda)
            images, labels = images.to(device), labels.to(device)
            f.write("准备数据：%0.3fMB\t" % (torch.cuda.memory_allocated() / 1000000))
            # images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            output = net(images)
            f.write("前向传播结束：%0.3fMB\t" % (torch.cuda.memory_allocated() / 1000000))
            # 输出为31922688字节bytes，输入images为（512，1，32，32）524288个浮点数
            loss = criterion(output, labels)
            f.write("计算损失函数：%0.3fMB\t" % (torch.cuda.memory_allocated() / 1000000))
            loss_list.append(loss.detach().cpu().item())
            batch_list.append(i + 1)

            # visdom可视化代码
            # if viz.check_connection():
            #     cur_batch_win = viz.line(torch.Tensor(loss_list), torch.Tensor(batch_list),
            #                              win=cur_batch_win, name='current_batch_loss',
            #                              update=(None if cur_batch_win is None else 'replace'),
            #                              opts=cur_batch_win_opts)

            loss.backward()  # 反向传播
            f.write("反向传播结束：%0.3fMB\t" % (torch.cuda.memory_allocated() / 1000000))
            optimizer.step()  # 梯度更新
            f.write("梯度更新：%0.3fMB\t" % (torch.cuda.memory_allocated() / 1000000))
            f.write("bw显存峰值:%0.3fMB\n" % (torch.cuda.max_memory_allocated() / 1000000))
            torch.cuda.reset_peak_memory_stats()

            if i % 10 == 0:
                f.write('Train - Epoch %d, Batch: %d, Loss: %f\n' % (epoch, i, loss.detach().cpu().item()))

        f.close()


def test():
    net.eval()  # 测试模式，不进行Dropout和数据标准化
    total_correct = 0
    avg_loss = 0.0

    for i, (images, labels) in enumerate(data_test_loader):
        images, labels = images.to(device), labels.to(device)
        # images, labels = images.cuda(), labels.cuda()

        output = net(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(data_test)
    with open(mem_log, "a+") as f:
        f.write('Test Avg. Loss: %f, Accuracy: %f\t' % (
            avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))
        f.close()


def train_and_test(epoch):
    with open(mem_log, "a+") as f:
        f.write(
            "%d台设备，当前使用设备为%s\n" % (torch.cuda.device_count(), torch.cuda.get_device_name(torch.cuda.current_device())))
        f.write("准备训练，当前分配显存:%0.3fMB\n" % (torch.cuda.memory_allocated() / 1000000))
        f.close()

    # 训练还未开始分配了259584字节
    train(epoch)
    with open(mem_log, "a+") as f:
        f.write("训练结束！！！\n")
        # f.write("显存峰值:%0.3fMB\n" % (torch.cuda.max_memory_allocated() / 1000000))
        f.write(torch.cuda.memory_summary())
        f.close()
    test()

    # 显存峰值输出为617894400字节

    dummy_input = torch.randn(1, 1, 32, 32, requires_grad=True)
    # print(type(dummy_input))
    dummy_input = dummy_input.to(device)

    torch.onnx.export(net, dummy_input, "lenet.onnx")

    onnx_model = onnx.load("lenet.onnx")
    onnx.checker.check_model(onnx_model)


def main():
    # if len(sys.argv) == 2:
    #     # for e in range(1, 16):
    #     #     train_and_test(e)
    #     train_and_test(1)
    # else:
    #     print("请输入日期以及本次实验目的：")

    train_and_test(1)


if __name__ == '__main__':
    main()
