import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


# 动态导入类的函数，用于加载配置文件中的类（如graph.ntu_rgb_d.Graph）
def import_class(name):
    components = name.split('.')  # 将类路径按点分割，如'model.ctrgcn.Model'
    mod = __import__(components[0])  # 导入顶级模块（如'model'）
    for comp in components[1:]:  # 逐层获取子模块或类
        mod = getattr(mod, comp)
    return mod  # 返回最终的类对象


# 初始化多分支卷积的权重和偏置，使用正态分布初始化权重，常数0初始化偏置
def conv_branch_init(conv, branches):
    weight = conv.weight  # 获取卷积层的权重张量
    n = weight.size(0)  # 输出通道数
    k1 = weight.size(1)  # 输入通道数
    k2 = weight.size(2)  # 卷积核高度
    # 使用He初始化变体，考虑分支数调整方差
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)  # 偏置初始化为0


# 初始化卷积层的权重（Kaiming）和偏置（0）
def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')  # Kaiming初始化，适用于卷积
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)  # 偏置初始化为0


# 初始化BatchNorm层的权重和偏置
def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)  # 设置缩放因子（如1）
    nn.init.constant_(bn.bias, 0)  # 设置偏移为0


# 时间卷积单元（TCN），用于处理时间序列
class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)  # 计算填充，确保输出时间维度不变
        # 定义2D卷积，处理(N, C, T, V)输入，仅在时间维度(T)上卷积
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)  # 批归一化，稳定训练
        self.relu = nn.ReLU(inplace=True)  # ReLU激活，inplace=True节省内存
        conv_init(self.conv)  # 初始化卷积权重和偏置
        bn_init(self.bn, 1)  # 初始化BN权重为1，偏置为0

    def forward(self, x):
        x = self.bn(self.conv(x))  # 卷积后归一化
        return x  # 返回结果，不应用ReLU（由外部决定）


# 图卷积单元（GCN），处理骨架的空间关系
class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=True):
        super(unit_gcn, self).__init__()
        self.out_c = out_channels  # 输出通道数
        self.in_c = in_channels  # 输入通道数
        self.num_subset = A.shape[0]  # 子图数量（如3个拓扑结构）
        self.adaptive = adaptive  # 是否自适应调整邻接矩阵

        # 自适应邻接矩阵，若启用则为可训练参数，否则为固定值
        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)

        # 为每个子图定义独立的卷积层
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))  # 1x1卷积调整通道

        # 残差连接，若输入输出通道不同，添加降采样
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x  # 通道一致时直接返回输入

        self.bn = nn.BatchNorm2d(out_channels)  # 输出归一化
        self.relu = nn.ReLU(inplace=True)  # ReLU激活

        # 初始化所有卷积和BN层
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)  # BN初始化，权重接近0
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)  # 每个分支卷积单独初始化

    # 对邻接矩阵进行L2归一化，增强稳定性
    def L2_norm(self, A):
        A_norm = torch.norm(A, 2, dim=1, keepdim=True) + 1e-4  # N,1,V，防止除0
        A = A / A_norm
        return A

    def forward(self, x):
        N, C, T, V = x.size()  # 输入形状：(批量大小, 通道, 时间, 关节数)

        y = None
        if self.adaptive:
            A = self.PA  # 使用自适应邻接矩阵
            A = self.L2_norm(A)  # 归一化
        else:
            A = self.A.cuda(x.get_device())  # 固定邻接矩阵移到GPU

        # 对每个子图进行图卷积
        for i in range(self.num_subset):
            A1 = A[i]  # 第i个子图的邻接矩阵 (V, V)
            A2 = x.view(N, C * T, V)  # 展平时间和通道 (N, C*T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))  # 图卷积后恢复形状
            y = z + y if y is not None else z  # 累加结果

        y = self.bn(y)  # 归一化
        y += self.down(x)  # 残差连接
        y = self.relu(y)  # 激活
        return y


# TCN和GCN组合单元，时空特征提取
class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)  # 图卷积
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride)  # 时间卷积
        self.relu = nn.ReLU(inplace=True)

        # 残差连接设置
        if not residual:
            self.residual = lambda x: 0  # 无残差
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x  # 直接相加
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)  # 调整通道和步幅

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))  # GCN -> TCN -> 残差 -> ReLU
        return y


# 完整CTR-GCN模型
class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True, num_set=3):
        super(Model, self).__init__()

        # 动态加载图结构类
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        # 初始化邻接矩阵，3个单位矩阵表示初始拓扑
        A = np.stack([np.eye(num_point)] * num_set, axis=0)  # (3, 25, 25)
        self.num_class = num_class
        self.num_point = num_point
        # 输入数据归一化，处理(N, M*V*C, T)
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # 定义10层时空卷积网络，逐步增加通道数并降采样时间
        self.l1 = TCN_GCN_unit(3, 64, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(64, 64, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(64, 128, A, stride=2, adaptive=adaptive)  # 降采样T
        self.l6 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(128, 128, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(128, 256, A, stride=2, adaptive=adaptive)  # 再次降采样T
        self.l9 = TCN_GCN_unit(256, 256, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(256, 256, A, adaptive=adaptive)

        # 全连接层，将特征映射到类别
        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))  # 初始化FC权重
        bn_init(self.data_bn, 1)  # 初始化BN

        # Dropout可选，用于正则化
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        N, C, T, V, M = x.size()  # 输入：(批量, 通道, 时间, 关节, 人数)

        # 数据重排为(N, M*V*C, T)以应用BN
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        # 恢复形状为(N*M, C, T, V)以输入网络
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # 通过10层时空卷积
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # 池化：(N*M, C, T, V) -> (N, C)
        c_new = x.size(1)  # 通道数（如256）
        x = x.view(N, M, c_new, -1)  # (N, M, C, T*V)
        x = x.mean(3).mean(1)  # 平均池化，沿T*V和M维度
        x = self.drop_out(x)  # 应用Dropout（若启用）

        return self.fc(x)  # 输出分类结果 (N, num_class)