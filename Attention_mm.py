import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def pause():
    programPause = input("Press the <ENTER> key to continue...")

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, num_heads, dim_per_head, attention_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_per_head = dim_per_head

    def forward(self, q, k, v, dataset, rpn_shape,  scale=None, attn_mask=None):
        if len(rpn_shape) ==4 :
           bz = rpn_shape[0]*rpn_shape[1]
        elif len(rpn_shape) ==5 :
            bz = rpn_shape[0]*rpn_shape[1]*rpn_shape[2]
        else:
           bz = rpn_shape[0] 
        k = k.view(bz, -1, self.num_heads, self.dim_per_head).transpose(1,2)
        v = v.view(bz, -1, self.num_heads, self.dim_per_head).transpose(1,2)
        q = q.view(bz, -1, self.num_heads, self.dim_per_head).transpose(1,2)
             
        attention = torch.matmul(q, k.transpose(-2, -1)).mul(scale)
        if attn_mask is not None:
            attention = attention.masked_fill(attn_mask, -1e9)

        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, v)

        return context.transpose(1,2).contiguous(), attention#
    
class MultiHeadAttention(nn.Module):

    def __init__(self, input_dim, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads

        self.num_heads = num_heads

        self.linear_k = nn.Linear(input_dim, model_dim)
        self.linear_v = nn.Linear(input_dim, model_dim)
        self.linear_q = nn.Linear(input_dim, model_dim)

        nn.init.xavier_uniform_(self.linear_k.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.linear_v.weight, gain=nn.init.calculate_gain('relu')) 
        nn.init.xavier_uniform_(self.linear_q.weight, gain=nn.init.calculate_gain('relu'))
        
        self.model_dim = model_dim      
        self.dot_product_attention = ScaledDotProductAttention(self.num_heads, self.dim_per_head, 0.0)
            
    def forward(self, query_i, dataset, is_training=False, attn_mask=None):

        query_l = self.linear_q(query_i)
        value_l =   self.linear_v(query_i)
        key_l =    self.linear_k(query_i)

        rpn_shape = query_l.size()
             
        scale = (self.model_dim // self.num_heads) ** -0.5
        context, _ = self.dot_product_attention(
            query_l, key_l, value_l, dataset, rpn_shape, scale,  attn_mask=None)
        output_context = context.view(rpn_shape)

        return query_l, output_context
                                      
