import os
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm

from model import vgg


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=0)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.next()

    model_name = "vgg16"
    net = vgg(model_name=model_name, num_classes=5, init_weights=True)
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    epochs = 30
    # 设置一个参数来更新历史最高精确率
    best_acc = 0.0
    save_path = './{}Net.pth'.format(model_name)
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        # 每一轮epoch训练都是把running_loss损失累加变量重新置0
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        # 使用枚举迭代训练集
        # step为枚举的索引
        # data为索引对应的图片和标签值
        for step, data in enumerate(train_bar):
            # 取出训练集的图片和标签值
            images, labels = data
            # 清空梯度
            optimizer.zero_grad()
            # 将训练集输入网络
            outputs = net(images.to(device))
            # 计算训练损失
            loss = loss_function(outputs, labels.to(device))
            # 梯度反向传播
            loss.backward()
            # 参数更新
            optimizer.step()

            # print statistics
            # 损失累加
            running_loss += loss.item()

            # 打印训练轮数和本轮的训练损失
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        # 验证集验证代码
        # validate
        net.eval()
        # 设定一个参数来累计精确率，以便于后面求得平均精确率
        acc = 0.0  # accumulate accurate number / epoch
        # 使用torch.no_grad()在预测过程不会记录梯度
        with torch.no_grad():
            # 这个是运行时显示进度条的工具
            val_bar = tqdm(validate_loader)
            # 迭代数据集
            for val_data in val_bar:
                # 取出验证集的图片和标签值
                val_images, val_labels = val_data
                # 将验证集数据输入网络
                outputs = net(val_images.to(device))
                # output = torch.max(input, dim)
                # input是softmax函数输出的一个tensor
                # dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值
                predict_y = torch.max(outputs, dim=1)[1]
                # 累积精确率
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
        # 求本轮训练的平均精确率
        val_accurate = acc / val_num
        # 打印本轮训练结果的信息
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        # 更新最高精确率
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
