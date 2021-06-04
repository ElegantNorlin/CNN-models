import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 图片预处理操作函数
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    # 定义图片所在路径
    img_path = "../tulip.jpg"
    # 查看图片路径是否存在
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # 用PIL库读取图片
    img = Image.open(img_path)
    # 展示图片
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    # 打开定义的json文件
    json_file = open(json_path, "r")
    # class_indict字典
    class_indict = json.load(json_file)

    # 初始化网络模型
    # create model
    model = AlexNet(num_classes=5).to(device)

    # load model weights
    # 定义训练好的网络权重所在路径
    weights_path = "./AlexNet.pth"
    # 检查权重路径是否存在
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    # 读取网络权重
    model.load_state_dict(torch.load(weights_path))
    # 模型验证为了关闭dropout，因为这里要预测，所以不需要dropout，训练过程需要dropout
    model.eval()
    # 不跟踪参数的损失梯度
    with torch.no_grad():
        # predict class
        # 维度压缩
        output = torch.squeeze(model(img.to(device))).cpu()
        # 将预测结果放入softmax函数
        predict = torch.softmax(output, dim=0)
        # 预测结果（获取概率最大的结果所对应的索引值）
        predict_cla = torch.argmax(predict).numpy()
    # 打印预测出的类别名称一级预测概率
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()
