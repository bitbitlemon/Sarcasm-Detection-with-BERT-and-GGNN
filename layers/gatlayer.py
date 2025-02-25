# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer for Graph Attention Networks (GAT).
    This layer applies attention mechanisms to graph-based data.
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        """
        Initialize the Graph Attention Layer.

        :param in_features: Number of input features (dimensionality of input data).
        :param out_features: Number of output features (dimensionality of the output).
        :param dropout: Dropout rate for regularization.
        :param alpha: Leaky ReLU negative slope for the attention coefficients.
        :param concat: If True, concatenate the attention heads, otherwise take the mean.
        """
        super(GraphAttentionLayer, self).__init__()
        
        self.dropout = dropout  # Dropout rate for regularization
        self.in_features = in_features  # Input feature dimension
        self.out_features = out_features  # Output feature dimension
        self.alpha = alpha  # Negative slope for LeakyReLU
        self.concat = concat  # If True, concatenate attention heads

        # Weight matrix to transform input features
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)  # Xavier initialization

        # Attention coefficients matrix
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # Xavier initialization

        # LeakyReLU activation function for the attention scores
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        """
        Forward pass of the Graph Attention Layer.
        
        :param input: Input feature matrix (N x in_features), where N is the number of nodes.
        :param adj: Adjacency matrix (N x N) representing the graph structure.
        
        :return: Output feature matrix (N x out_features).
        """
        # Step 1: Linear transformation of the input features using the weight matrix
        h = torch.matmul(input, self.W)  # (N x in_features) * (in_features x out_features) -> (N x out_features)
        
        N = h.size()[0]  # Number of nodes in the graph

        # Step 2: Calculate attention scores for each pair of nodes
        # a_input is a concatenation of the node features for pairs of nodes
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1)  # (N * N) x (2 * out_features)
        a_input = a_input.view(N, -1, 2 * self.out_features)  # Reshape to (N x N, 2 * out_features)

        # Step 3: Apply attention mechanism (LeakyReLU activation)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # (N x N), attention scores

        # Step 4: Mask attention scores where the adjacency matrix is zero (no edge between nodes)
        zero_vec = -9e15 * torch.ones_like(e)  # Large negative number for masking
        attention = torch.where(adj > 0, e, zero_vec)  # Apply the adjacency matrix to mask
        attention = F.softmax(attention, dim=1)  # Normalize attention scores across neighbors

        # Step 5: Apply dropout to attention scores (for regularization)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # Step 6: Aggregate the neighbor features weighted by attention scores
        h_prime = torch.matmul(attention, h)  # (N x N) x (N x out_features) -> (N x out_features)

        # Step 7: Return the output with or without concatenation (based on concat flag)
        if self.concat:
            return F.elu(h_prime)  # Apply ELU activation if concatenating
        else:
            return h_prime  # Just return the attention-weighted feature matrix
