import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms


def main():
    # 定义数据预处理操作
    transform = transforms.Compose(
        # transforms.ToTensor()将数据变成向量
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))将数据做标准化处理
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    # root='./data'数据集要下载到哪里 
    # train=True导入训练集样本
    # transform=transform将数据集做transform中的预处理操作
    # download=False表示数据集要下载，当数据集下载完之后，我们要把download的值修改为False
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,  
                                             download=True, transform=transform)
    # num_workers=0这个参数windows下设置成别的参数可能会报错
    # batch_size=36每一个bacth中包含36个数据
    # shuffle=True打乱数据集
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=0)

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    # train=False时，导入测试集
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
                                             shuffle=False, num_workers=0)
    # 将数据集放到迭代器中
    val_data_iter = iter(val_loader)
    # val_image为测试集图片
    # val_label为测试集标签
    val_image, val_label = val_data_iter.next()
    # 标签列表 
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # 导入定义的LeNet类
    net = LeNet()
    # 定义损失函数
    loss_function = nn.CrossEntropyLoss()
    # 定义优化器
    # net.parameters()需要训练的参数列表
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # 迭代训练集（epoch为数据集迭代的次数）
    for epoch in range(5):  # loop over the dataset multiple times
        # 累加训练过程的损失,后面用来打印本轮训练过程的平均精度而设置的
        running_loss = 0.0
        # enumerate(train_loader, start=0)从索引0开始迭代train_loader数据集
        # step为索引号
        # data为对应索引数据集中的数据
        for step, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs为当前要训练的图片
            # labels为当前训练图片的真实标签值
            inputs, labels = data

            # zero the parameter gradients
            # 把模型中参数的梯度置为0
            optimizer.zero_grad()
            # forward + backward + optimize
            # 将输入输入到网络中得到输出
            outputs = net(inputs)
            # 损失函数loss_function(outputs, labels)
            # outputs为网络预测的结果
            # labels为数据的真是标签
            loss = loss_function(outputs, labels)
            # 损失函数反向传播
            loss.backward()
            # 优化器进行参数更新
            optimizer.step()

            # 信息打印的过程
            # print statistics
            # 累加到running_loss变量当中
            running_loss += loss.item()
            if step % 500 == 499:    # print every 500 mini-batches
                # torch.no_grad()被该语句管理部分的参数部位参与梯度计算
                with torch.no_grad():
                    outputs = net(val_image)  # [batch, 10]
                    # 寻找输出特征中最大值的索引是多少，dim=1的意思是我们跳过batch，从索引为1的维度开始寻找
                    # [1]我们只需要拿到最大值所在的索引，而不需要知道具体的数值
                    predict_y = torch.max(outputs, dim=1)[1]
                    # torch.eq(predict_y, val_label)将预测结果于数据的真是标签对比是否相等，返回1或0
                    # sum()加起来就知道在本次训练过程中预测对了多少个样本
                    accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

                    # running_loss / 500为平均训练误差
                    print('[%d, %5d] train_loss: %.3f  test_accuracy: %.3f' %
                          (epoch + 1, step + 1, running_loss / 500, accuracy))
                    # running_loss清零，准备进行下一轮数据集训练
                    running_loss = 0.0

    print('Finished Training')
    # 将模型权重保存到设置的路径下
    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()
