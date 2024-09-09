import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel

class BERTOnly(nn.Module):
    def __init__(self, bert, opt):
        super(BERTOnly, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        # 将输入维度从 100 改为 768，以匹配 BERT 的输出
        self.classifier = nn.Linear(768, opt.polarities_dim)  # opt.polarities_dim 应该是你要分类的类别数，例如 2 表示二分类

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        # 使用 BERT 提取特征
        _, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        # 使用全连接层进行分类
        output = self.classifier(pooled_output)
        return output

