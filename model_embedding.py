#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 19:51:49 2019

@author: jace-belere
"""
import torch
import torch.nn as nn
import numpy as np
import time


class ModelEmbeddings(nn.Module):
  """
    We creat this class for future work on better embedding.
  """
  def __init__(self,vocab_size,embed_size,pad_token_id=0):
    super(ModelEmbeddings,self).__init__()
    
    self.vocab_size=vocab_size
    self.embed_size=embed_size
    self.token2id={}
    self.token2id['<pad>']=0
    self.token2id['<s>']=1
    self.token2id['<unknow>']=2
#    self.device=device
    self.recurrent_layer=nn.GRU(embed_size,embed_size,num_layers=2,batch_first=True,bidirectional=True)
    self.affine=nn.Linear(2*embed_size,embed_size)
    self.embedding=nn.Embedding(vocab_size+len(self.token2id),embed_size,pad_token_id)
    
    
  def forward(self,batch_data,device=torch.device('cpu')):
    """
      seqence: shape: (batch,seq_len) --> long tensor
    """
#    tic=time.time()
    batch_data=torch.tensor([ [self.token2id.get(j,self.token2id['<unknow>']) for j in i] for i in batch_data],device=device)
#    print("use {0} seconds for converting batch".format(time.time()-tic))
    
    return  self.affine(self.recurrent_layer(self.embedding(batch_data))[0])
  
#  def to_input_tensor(batch_data, device):
#    """
#      batch_data: (batch_size,maxSenLen) shape str list
#    """
  
  
  @staticmethod
  def load_from_file(vec_file,pad_token_id=0):
    """
      Create the complete word embedding from word2vec output file.
    """
    print(F"loading pre-train embedding file: {vec_file}...")
    to_read=open(vec_file,'r')
    line=to_read.readline()
    vocab_size,embed_size=np.array(line.split(" "),dtype=int)
    instance=ModelEmbeddings(int(vocab_size),int(embed_size),pad_token_id)
    pretrain_weight=torch.zeros(instance.embedding.weight.size())
    while(True):
      line=to_read.readline()
      if(not line): break
      l=line.split(' ')
#      print(l)
#      if(l[0]==)
      pretrain_weight[len(instance.token2id)]=torch.from_numpy(np.array(l[1:-1],dtype='float64'))
      instance.token2id[l[0]]=len(instance.token2id)
    instance.embedding=nn.Embedding.from_pretrained(pretrain_weight)
    print("Done embedding load!")
#    instance.embedding.weight.require_grad=False
    return instance
    
#Mini Test
cls_embed=ModelEmbeddings(10,5,0)
in_words=[['1','2','3','4']]

print(cls_embed(in_words).size())