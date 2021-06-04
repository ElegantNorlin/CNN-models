import os
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import AlexNet


def main():
    # 如果当前有适合cuda的gpu时就使用cuda0，如果没有则使用cpu设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # 定义数据集预处理字典
    data_transform = {
    # RandomResizedCrop随机裁剪
    # RandomHorizontalFlip随机翻转
    # 将数据转换成向量
    # 标准化处理
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224 , 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
    # 获取相应路径
    # data_root = os.path.abspath(os.path.join(os.getcwd()))为获取当前目录，这个根据自己路径设置写参数
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    # 获取数据集所在路径
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    # 判断我们获取的数据集路径是否真实存在
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    # 获取训练集，并用定义的预处理函数做预处理
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    # 获取训练集图片的数量
    train_num = len(train_dataset)
    
    # 获取分类名对应的索引
    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    # 将flower_list字典的“键”与“值”的位置交换位置
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # 将分类列表写成json格式
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    # 保存刚刚写的json文件
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # 设置batch_size每一个batch32张训练图片
    batch_size = 32
    # linux系统下跑此程序可以把num_workers值设置为nw，windwos下num_workers值只能为0，不然会报错
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    # print('Using {} dataloader workers every process'.format(nw))

    # 加载训练集并打乱顺序
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    # 获取验证集图片的数量
    val_num = len(validate_dataset)
    # 加载验证集并打乱顺序
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=0)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()
    #
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    # 加载定义的AlexNet模型类并传入参数
    net = AlexNet(num_classes=5, init_weights=True)
    # 指定训练所用设备
    net.to(device)
    # 设置损失函数交叉熵损失函数
    loss_function = nn.CrossEntropyLoss()
    # 下面的para列表可以查看模型的参数，如果比较好奇可以跑起来调试看一下模型的参数
    # para = list(net.parameters())
    # 定义优化器，优化网络模型的参数
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    # 定义
    epochs = 10
    # 定义网络模型存放的路径
    save_path = './AlexNet.pth'
    # 设置验证集准确率为0，如果验证集验证有更高的准确率，那么将会更新此参数
    best_acc = 0.0
    # train_steps
    train_steps = len(train_loader) 
    print("train_steps=%d" %train_steps)
    for epoch in range(epochs):
        # train
        # 开始训练
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            # 回去图片及其对应的标签值
            images, labels = data
            # 清空梯度
            optimizer.zero_grad()
            # 指定训练集由什么设备去训练
            outputs = net(images.to(device))
            # 将预测值和标签值进行损失函数计算
            loss = loss_function(outputs, labels.to(device))
            # 损失函数反向传播
            loss.backward()
            # 损失函数参数更新
            optimizer.step()

            # print statistics
            # 累加损失
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate验证
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        # torch.no_grad()表示不参数梯度不发生变化，验证过程模型参数梯度不应该被更改
        with torch.no_grad():
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            # 更新历史最高精度
            best_acc = val_accurate
            # 保存权重文件
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
