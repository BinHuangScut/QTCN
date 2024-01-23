import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import attention_layer
from attention_layer import MultiheadAttention
from tcn import Chomp1d, TemporalBlock, TemporalConvNet

class MultiTaskBlock(nn.Module):
    def __init__(self, input_size):
        super(MultiTaskBlock, self).__init__()
        self.linear_1 = nn.Linear(input_size, 128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_3 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        return x

# Multi_task_Model with attention layer
class Attention_TCN_Model(nn.Module):
    # def __init__(self, attention_input_size, attention_embed_dim, num_heads, lstm_hidden_size, lstm_num_layers, num_loads):
    def __init__(self, attention_input_size, attention_embed_dim, num_heads, tcn_hidden_size, num_loads, num_blocks, num_weather):
        super().__init__()
        
        # parameter of attention
        self.attention_input_size = attention_input_size
        self.attention_embed_dim = attention_embed_dim
        self.num_heads = num_heads
        
        # parameter of tcn
        self.num_loads = num_loads
        self.tcn_hidden_size = tcn_hidden_size
        
        self.num_blocks = num_blocks
        self.num_weather = num_weather
        

        self.attention_1 = MultiheadAttention(input_dim = self.attention_input_size, embed_dim = self.attention_embed_dim, \
                                    num_heads = self.num_heads)
        self.attention_2 = MultiheadAttention(input_dim = self.attention_input_size, embed_dim = self.attention_embed_dim, \
                                    num_heads = self.num_heads)
        self.attention_3 = MultiheadAttention(input_dim = self.attention_input_size, embed_dim = self.attention_embed_dim, \
                                    num_heads = self.num_heads)

        
        self.TCN = TemporalConvNet(num_inputs = self.attention_embed_dim*self.num_loads+self.num_weather, num_channels = [self.tcn_hidden_size])
        self.blocks = nn.ModuleList([MultiTaskBlock(self.tcn_hidden_size) for _ in range(self.num_blocks)])
        
        
        
    def forward(self, x, x_weather):        
        # x: (batch_size, num_load, t_step)
        # attention layer input: batch_size, t_step, input_dim
        
        attention_1 = self.attention_1(x[:,:,0].unsqueeze(-1))
        attention_output_1 = attention_1[0].view(x.size(0), 1, -1)
        attention_score_1 = attention_1[1]

        attention_2 = self.attention_2(x[:,:,1].unsqueeze(-1))
        attention_output_2 = attention_2[0].view(x.size(0), 1, -1)
        attention_score_2 = attention_2[1]
        
        attention_3 = self.attention_3(x[:,:,2].unsqueeze(-1))
        attention_output_3 = attention_3[0].view(x.size(0), 1, -1)
        attention_score_3 = attention_3[1]
        
        tcn_input = torch.cat((attention_output_1, attention_output_2, attention_output_3), 1).transpose(1,2)
        tcn_input = torch.cat((x_weather, tcn_input), 1)
        TCN_output = self.TCN(tcn_input)
        
        linear_input = TCN_output[:, :, -1] 
        
        block_outputs = []
        for block in self.blocks:
            x = block(linear_input)
            block_outputs.append(x)
        
        block_outputs.append(attention_score_1)
        block_outputs.append(attention_score_2)
        block_outputs.append(attention_score_3)
        
        return block_outputs