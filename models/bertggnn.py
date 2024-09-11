# models/bertggnn.py

import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel
from layers.ggnnlayer import GGNNLayer


class BERTGGNN(nn.Module):
    def __init__(self, bert, opt):
        super(BERTGGNN, self).__init__()
        self.bert = bert

        # 初始化 GGNN 层时，传入多头注意力机制的 num_heads 参数
        self.ggnn = GGNNLayer(opt.bert_dim, opt.hidden_dim, num_heads=opt.num_heads)
        self.fc = nn.Linear(opt.hidden_dim, opt.polarities_dim)
        self.dropout = nn.Dropout(opt.dropout)

    def forward(self, inputs):
        text_bert_indices, bert_segments_indices, dependency_graph = inputs
        bert_out, _ = self.bert(text_bert_indices, token_type_ids=bert_segments_indices,
                                output_all_encoded_layers=False)
        ggnn_out = self.ggnn(bert_out, dependency_graph)
        ggnn_out = self.dropout(ggnn_out)
        logits = self.fc(ggnn_out[:, 0, :])  # 使用 [CLS] token 的表示
        return logits
