# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolutionLayer  # Importing a custom graph convolution layer

class BERTGCN(nn.Module):
    def __init__(self, bert, opt):
        """
        Initialize the BERTGCN model.
        
        :param bert: Pretrained BERT model
        :param opt: Hyperparameters and configuration options
        """
        super(BERTGCN, self).__init__()
        self.opt = opt  # Store options (like dimensions, dropout, etc.)
        self.bert = bert  # Pretrained BERT model

        # Define multiple graph convolution layers (GCN)
        self.gc1 = GraphConvolutionLayer(opt.bert_dim, opt.bert_dim)  # GCN layer 1
        self.gc2 = GraphConvolutionLayer(opt.bert_dim, opt.bert_dim)  # GCN layer 2
        self.gc3 = GraphConvolutionLayer(opt.bert_dim, opt.bert_dim)  # GCN layer 3
        self.gc4 = GraphConvolutionLayer(opt.bert_dim, opt.bert_dim)  # GCN layer 4
        # You can add more layers if needed (commented-out layers below)

        # Fully connected layer for classification (output size is determined by polarities_dim)
        self.fc = nn.Linear(opt.bert_dim, opt.polarities_dim)

        # Dropout layer for text embedding to prevent overfitting
        self.text_embed_dropout = nn.Dropout(0.3)

    def forward(self, inputs):
        """
        Forward pass of the BERTGCN model.
        
        :param inputs: A tuple containing:
            - text_bert_indices: Input token indices for BERT
            - bert_segments_ids: Segment IDs for BERT (to distinguish sentence A from B)
            - dependency_graph: Dependency graph for GCN
            - affective_graph: Affective graph for GCN
        :return: Output of the model (classification logits)
        """
        # Unpack the inputs
        text_bert_indices, bert_segments_ids, dependency_graph, affective_graph = inputs

        # Get BERT embeddings (we only use the last hidden state)
        text_out, _ = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, output_all_encoded_layers=False)

        # Apply GCN layers sequentially
        x = F.relu(self.gc1(text_out, dependency_graph))  # GCN layer 1
        x = F.relu(self.gc2(x, affective_graph))  # GCN layer 2
        x = F.relu(self.gc3(text_out, dependency_graph))  # GCN layer 3
        x = F.relu(self.gc4(x, affective_graph))  # GCN layer 4
        # If more layers are added, you can use similar pattern
        # x = F.relu(self.gc5(text_out, dependency_graph))  # GCN layer 5
        # x = F.relu(self.gc6(x, affective_graph))  # GCN layer 6

        # Attention mechanism: Compute similarity between GCN output and BERT output
        alpha_mat = torch.matmul(x, text_out.transpose(1, 2))  # Compute similarity score matrix
        alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)  # Softmax for attention weights

        # Apply attention to get weighted sum of BERT outputs
        x = torch.matmul(alpha, text_out).squeeze(1)  # Weighted sum of BERT outputs

        # Feed through a fully connected layer for classification
        output = self.fc(x)

        return output
