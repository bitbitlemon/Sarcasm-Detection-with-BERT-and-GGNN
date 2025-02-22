# layers/ggnnlayer.py

import torch
import torch.nn as nn


class GGNNLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=8, bias=True):
        super(GGNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads

        # Defines a weight matrix for message propagation
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))

        # Multi-head attention mechanism
        self.multihead_attn = nn.MultiheadAttention(embed_dim=out_features, num_heads=num_heads)

        # Define the GRU unit, which is used to update the status of the node
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

        # Compute the hidden layer of message propagation
        hidden = torch.matmul(text, self.weight)

        # Computes messages for neighbors and propagates them
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        aggregated = torch.matmul(adj, hidden) / denom

        if self.bias is not None:
            aggregated = aggregated + self.bias

        # Input and aggregated messages are passed through a multi-head attention mechanism
        attn_output, _ = self.multihead_attn(aggregated, aggregated, aggregated)

        # The status of the input and aggregated messages is updated through the GRU unit
        output = self.gru(attn_output.view(-1, self.out_features), text.view(-1, self.out_features))

        # Restore the original batch dimension
        output = output.view(text.size(0), text.size(1), self.out_features)

        return output
