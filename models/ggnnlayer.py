# layers/ggnnlayer.py

import torch
import torch.nn as nn

class GGNNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GGNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 定义权重矩阵，用于消息传播
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # 定义GRU单元，用于更新节点的状态
        self.gru = nn.GRUCell(out_features, out_features)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, text, adj):
        text = text.to(torch.float32)
        # 计算消息传播的隐藏层
        hidden = torch.matmul(text, self.weight)

        # 计算邻居的消息并传播
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        aggregated = torch.matmul(adj, hidden) / denom

        if self.bias is not None:
            aggregated = aggregated + self.bias

        # 将输入和聚合后的消息通过 GRU 单元更新状态
        output = self.gru(aggregated.view(-1, self.out_features), text.view(-1, self.out_features))

        # 恢复原始的 batch 维度
        output = output.view(text.size(0), text.size(1), self.out_features)

        return output
