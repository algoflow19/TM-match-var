#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:27:22 2019

@author: jace-belere
"""

import torch 
import torch.nn as nn



class textCNN(nn.Module):
  """
    Augment version for textCNN, we use two layers CNN network for long and short depend relation.
  """
  def __init__(self,embedding_size,cls_num,l1_channels_num=64,l2_channels_num=128):
    super(textCNN,self).__init__()
    self.filter_sizes=[2,3,4]
    self.filter_sizes2=[3,4,2]
    self.pool_sizes=[(2,2),(3,3),(2,2)] # (kernel_size,stride) for each tuple.
    self.l1_channels_num=l1_channels_num
    self.l2_channels_num=l2_channels_num
    self.embedding_size=embedding_size
    self.cls_num=cls_num
    self.lastScore=0
    self.l1_convs=nn.ModuleList([nn.Conv1d(embedding_size,l1_channels_num,i) for i in self.filter_sizes] )
    self.pools=nn.ModuleList([nn.MaxPool1d(i[0],stride=i[1]) for i in self.pool_sizes])
    self.l2_convs=nn.ModuleList([nn.Conv1d(l1_channels_num,l2_channels_num,i) for i in self.filter_sizes2])
    self.dropout = nn.Dropout(0.32)
    self.fc=nn.Linear(len(self.filter_sizes2)*l2_channels_num,cls_num)
    
#    self.pool3=nn.MaxPool1d(3,3)
#    self.pool2=nn.MaxPool1d(2,2)
    
  def forward(self,input_tensor: torch.tensor): 
    """
      input_tensor: shape(batch,maxLenSen,embed_size)
    """
    input_tensor=input_tensor.permute(0,2,1).contiguous()
    l1_outs=[conv(input_tensor) for conv in self.l1_convs]
    l2_outs=[  self.l2_convs[i](self.pools[i](l1_outs[i]))  for i in range(len(self.filter_sizes2))]
    l2_outs=[torch.max(out,2)[0] for out in l2_outs]
    l2_outs=self.dropout(torch.cat(l2_outs,1))
    
    
    return self.fc(l2_outs)
  

# Mini Test
batch=64
l=500
embed=200
c=27
a=torch.rand(batch,l,embed)
tc=textCNN(embed,c)
out=tc(a)
