import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel

class BERTOnly(nn.Module):
    def __init__(self, bert, opt):
        super(BERTOnly, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        # Change the input dimension from 100 to 768 to match Bert's output
        self.classifier = nn.Linear(768, opt.polarities_dim)  # opt.polarities_dim

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        # Extracting features using Bert
        _, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        # Classification using the fully connected layer
        output = self.classifier(pooled_output)
        return output

