import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1

class MultiScale_TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, dilations=[1,2], residual=True, residual_kernel_size=1):
        super(MultiScale_TemporalConv, self).__init__()
        num_branches = len(dilations) + 2  # 扩张分支 + MaxPool分支 + 1x1分支
        assert out_channels % num_branches == 0, f'# out channels ({out_channels}) must be divisible by # branches ({num_branches})'
        branch_channels = out_channels // num_branches
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=(kernel_size, 1),
                          padding=(int((kernel_size-1)/2)*dilation, 0), dilation=(dilation, 1), stride=(stride, 1)),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True)
            ))
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)
        ))
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
            for m in self.residual.modules():
                if isinstance(m, nn.Conv2d):
                    conv_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    bn_init(m, 1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = []
        for branch in self.branches:
            out.append(branch(x))
        out = torch.cat(out, dim=1)
        out += self.residual(x)
        out = self.bn(out)
        return out

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super(unit_tcn, self).__init__()
        pad = 0  # kernel_size=1 无需填充
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1),
                              padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class new_unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=True, attention=True):
        super(new_unit_gcn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = A.shape[0]
        self.adaptive = adaptive
        self.attention = attention
        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
            self.alpha = nn.Parameter(torch.zeros(1))
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if attention:
            num_jpts = A.shape[-1]
            ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
            pad = (ker_jpt - 1) // 2
            self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)
            self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
            rr = 16
            self.fc1c = nn.Linear(out_channels, out_channels // rr)
            self.fc2c = nn.Linear(out_channels // rr, out_channels)
            self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        A = self.PA if self.adaptive else A
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i] if self.adaptive else A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        if self.attention:
            se = y.mean(-2)  # N, C, V
            se1 = self.sigmoid(self.conv_sa(se))
            # y = y * se1.unsqueeze(-2) + y
            y = y * se1.unsqueeze(-2) # 修正为仅乘法
            se = y.mean(-1)  # N, C, T
            se1 = self.sigmoid(self.conv_ta(se))
            # y = y * se1.unsqueeze(-1) + y
            y = y * se1.unsqueeze(-1) # 修正为仅乘法
            se = y.mean(-1).mean(-1)  # N, C
            se1 = self.relu(self.fc1c(se))
            se2 = self.sigmoid(self.fc2c(se1))
            # y = y * se2.unsqueeze(-1).unsqueeze(-1) + y
            y = y * se2.unsqueeze(-1).unsqueeze(-1) # 修正为仅乘法
        return y

class new_TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, attention=True):
        super(new_TCN_GCN_unit, self).__init__()
        self.gcn1 = new_unit_gcn(in_channels, out_channels, A, adaptive=adaptive, attention=attention)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=5, stride=stride, dilations=[1,2], residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
            for m in self.residual.modules():
                if isinstance(m, nn.Conv2d):
                    conv_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    bn_init(m, 1)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y

class ACT_GCN(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3, drop_out=0, adaptive=True):
        super(ACT_GCN, self).__init__()
        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
        A = self.graph.A
        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        self.l1 = new_TCN_GCN_unit(in_channels, 64, A, residual=False, adaptive=adaptive, attention=True)
        self.l2 = new_TCN_GCN_unit(64, 64, A, adaptive=adaptive, attention=True)
        self.l3 = new_TCN_GCN_unit(64, 64, A, adaptive=adaptive, attention=True)
        self.l4 = new_TCN_GCN_unit(64, 64, A, adaptive=adaptive, attention=True)
        self.l5 = new_TCN_GCN_unit(64, 128, A, stride=2, adaptive=adaptive, attention=True)
        self.l6 = new_TCN_GCN_unit(128, 128, A, adaptive=adaptive, attention=True)
        self.l7 = new_TCN_GCN_unit(128, 128, A, adaptive=adaptive, attention=True)
        self.l8 = new_TCN_GCN_unit(128, 256, A, stride=2, adaptive=adaptive, attention=True)
        self.l9 = new_TCN_GCN_unit(256, 256, A, adaptive=adaptive, attention=True)
        self.l10 = new_TCN_GCN_unit(256, 256, A, adaptive=adaptive, attention=True)
        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
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
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)
        return self.fc(x)