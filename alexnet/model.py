import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        # 卷积层的定义，具体的输入输出特征尺寸计算公式可以自行学习其他资料，利用好网络这把双刃剑
        '''
                    nn.Conv2d(padding=2)其中的padding接收的参数可以是整型也可以是元组
                    如果是整形，那么相当于再图片的四周补上padding行0
                    如果是元组(a,b):
                        在图片的上下填充a行0
                        在图片的左右填充b列0
                    '''
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[96, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[96, 27, 27]
            nn.Conv2d(96, 256, kernel_size=5, stride=1,padding=2),           # output[256, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[256, 13, 13]
            nn.Conv2d(256, 384, kernel_size=3,stride=1, padding=1),          # output[384, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3,stride=1,padding=1),          # output[384, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3,stride=1, padding=1),          # output[256, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[256, 6, 6]
        )
        # 全连接层的定义
        self.classifier = nn.Sequential(
            # p=0.5代表神经元随机丢弃(失活)的比例
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        # 是否初始化权重
        if init_weights:
            # 调用初始化权重函数
            self._initialize_weights()
    # Alexnet网络的前向传播过程
    def forward(self, x):
        # 卷积层
        x = self.features(x)
        # 特征拉直，然后输入全连接层
        '''
        假设类型为 torch.tensor 的张量 t 的形状如下所示：(2,4,3,5,6)，则 orch.flatten(t, 1, 3).shape 的结果为
        (2, 60, 6)。将索引为 start_dim 和 end_dim 之间（包括该位置）的数量相乘，其余位置不变。
        因为默认 start_dim=0，end_dim=-1，所以 torch.flatten(t) 返回只有一维的数据。
        '''
        x = torch.flatten(x, start_dim=1)
        # 将拉直的特征输入全连接层
        x = self.classifier(x)
        return x
    # 初始化权重函数
    def _initialize_weights(self):
        # self.modules()是一个迭代器
        # 这里for循环是迭代模型中的对象
        for m in self.modules():
            # 判断对象是否是卷积层
            if isinstance(m   , nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 如果偏置项不为空，那么则
                if m.bias is not None:
                    # m.bias初始化为0
                    nn.init.constant_(m.bias, 0)
            # 判断对象是否是全连接层
            elif isinstance(m, nn.Linear):
                # 按照正态分布给权重初始化
                nn.init.normal_(m.weight, 0, 0.01)
                # m.bias初始化为0
                nn.init.constant_(m.bias, 0)
