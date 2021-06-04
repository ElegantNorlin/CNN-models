import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
# 引入定义的vgg类
from model import vgg


def main():
    # 使用什么设备进行训练
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 定义数据集预处理操作函数
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    # 添加预测图片的路径
    img_path = "../tulip.jpg"
    # 验证图片路径是否存在
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # 根据路径，打开图片
    img = Image.open(img_path)
    # 展示图片
    plt.imshow(img)
    # [N, C, H, W]
    # 调整图片维度顺序
    img = data_transform(img)
    # expand batch dimension扩展数据维度
    # 这个函数可以看一下这个链接，讲的很清楚https://zhuanlan.zhihu.com/p/86763381
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    
    # create model
    model = vgg(model_name="vgg16", num_classes=5).to(device)
    # load model weights
    weights_path = "./vgg16Net.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()
