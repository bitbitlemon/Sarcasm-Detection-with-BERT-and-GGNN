# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer  # Custom graph attention layer

class BERTGAT(nn.Module):
    """
    BERT with Graph Attention Networks (GAT) for graph-based learning.
    """
    def __init__(self, bert, num_features):
        """
        Initialize the BERTGAT model.
        
        :param bert: Pretrained BERT model
        :param num_features: Number of features in the graph (i.e., input size for GraphAttentionLayer)
        """
        super(BERTGAT, self).__init__()
        
        self.bert = bert  # Pretrained BERT model
        self.dropout = 0.6  # Dropout rate to prevent overfitting
        self.num_hidden = 8  # Number of hidden units in each attention layer
        self.alpha = 0.2  # Negative slope for leaky ReLU
        self.att_head = 8  # Number of attention heads in multi-head attention
        self.num_class = 2  # Number of output classes (polarities_dim)

        # Define multiple Graph Attention Layers (one for each attention head)
        self.attentions = [ 
            GraphAttentionLayer(num_features,
                                self.num_hidden,
                                dropout=self.dropout,
                                alpha=self.alpha,
                                concat=True)  # Concat attention heads
            for _ in range(self.att_head)
        ]
        
        # Add each attention layer as a module
        for i, attention in enumerate(self.attentions):
            self.add_module(f'attention_{i}', attention)

        # Final output attention layer, no concatenation here
        self.out_att = GraphAttentionLayer(self.num_hidden * self.att_head, self.num_class, 
                                           dropout=self.dropout, alpha=self.alpha, concat=False)

    def forward(self, inputs):
        """
        Forward pass of the BERTGAT model.
        
        :param inputs: A tuple containing:
            - text_bert_indices: Token indices for BERT model
            - bert_segments_ids: Segment IDs for BERT model (A or B)
            - dependency_graph: Dependency graph for GAT
            - sdat_graph: Second graph (for additional graph structure)
        
        :return: Log probabilities of the classes
        """
        text_bert_indices, bert_segments_ids, dependency_graph, sdat_graph = inputs
        
        # Get BERT embeddings (we only use the last hidden state)
        x, _ = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, output_all_encoded_layers=False)
        
        # Apply dropout to BERT output embeddings
        x = F.dropout(x, self.dropout)

        # Apply graph attention on BERT output using multiple attention heads
        # Concatenate the results from all attention heads
        x = torch.cat([att(x, dependency_graph) for att in self.attentions], dim=1)
        
        # Apply dropout again after concatenation
        x = F.dropout(x, self.dropout)
        
        # Apply the final graph attention layer to the concatenated attention output
        x = F.elu(self.out_att(x, sdat_graph))  # ELU activation after final attention layer
        
        # Return log softmax of the output for multi-class classification
        return F.log_softmax(x, dim=1)
