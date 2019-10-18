#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 20:23:52 2019

@author: jace-belere
"""
import matplotlib.pyplot as plt
__maxLen__=20 #k

def EDA_len(corpus_file,__maxLen__=20):
  y=[0]*__maxLen__
  x=[i for i in range(0,__maxLen__)]
  cf=open(corpus_file,'r')
  for line in cf:
#    print(len(line.split(' '))//1000)
    y[len(line.split(' '))//1000]+=1
  plt.bar(x,y)
  cf.close()
  print(y)
#  return y

#y=[188962, 77983, 20933, 6332, 2274, 1124, 678, 450, 354, 239, 174, 109, 102, 64, 43, 36, 19, 13, 6, 12, 7, 1, 4, 7,
#   2, 2, 0, 2, 4, 1, 0, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
#   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#x=[]
#for i in range(1,len(y)+1):
#  x.append(i*y[i-1])
#print(x)

def EDA_cls(labels_file,cls_num=28):
  y=[0]*cls_num
  x=[i for i in range(0,cls_num)]
  lf=open(labels_file,'r')
  lf.readline()
  for line in lf:
#    print(line)
    y[int(line.split(',')[1])]+=1
  plt.bar(x,y)
  lf.close()
  print(y)
  return y

#EDA('corpus.txt',100)
#EDA('')
#
#[0,
# 16220,
# 8842,
# 10294,
# 4492,
# 12104,
# 13281,
# 4022,
# 22174,
# 13398,
# 8266,
# 8167,
# 4208,
# 23813,
# 6662,
# 4623,
# 4403,
# 4801,
# 7149,
# 5190,
# 16569,
# 1109,
# 28,
# 112,
# 31,
# 2,
# 1,
# 1]

#==============================================================================

import numpy as np
import torch
from torch.utils import data

__CURRENT_TRAIN_DATASIZE__=199962
__CURRENT_TEST_DATASIZE__=99983
# 32768

class DataSet():
  """
    There is a important observation that all sentenses in corpus are start and end with certain word,
    so there is no need for start and end token!
    
    This class read corpus file and dircetly produce batch input tensor for our model.
  """
  def __init__(self,src_file,tgt_file=None,batch_size=32,maxSenLen=3072,wantLine=2):
    self.batch_size=batch_size
    self.maxSenLen=maxSenLen
    self.src_file=src_file
    self.tgt_file=tgt_file
    self.dataset_size=0
    self.train_batchs=[]
    self.indic=0
    src=open(src_file,'r')
    src.readline()
    tgt=None
    self.tgt_batchs=None
    if(tgt_file is not None):
      tgt=open(tgt_file,'r')
      tgt.readline()
      self.tgt_batchs=[]
    print('loading data from {0}...'.format(src_file))
    while(True):
      line=src.readline()
      if(line==''): break
      self.dataset_size+=1
      self.train_batchs.append(line.split(',')[wantLine].split(' ')[:maxSenLen])
      if(self.train_batchs[-1][-1]=='\n'): self.train_batchs[-1].pop()
    print('Done loading.')
    src.close()
#    train_batchs=np.asarray(train_batchs,dtype='float32')
#    train_batchs=torch.from_numpy(train_batchs)
    if(tgt):
      self.tgt_batchs=[0]*self.dataset_size
      print('loading labels...')
      for i in range(0,self.dataset_size):
        self.tgt_batchs[i]=int(tgt.readline().split(',')[1])-1
#        if(self.tgt_batchs[i][-1]=='\n'): self.tgt_batchs[-1].pop()
      print('Done label loading.')
      tgt.close()
    
#      tgt_batchs=np.asarray(tgt_batchs,dtype='int')
#      tgt_batchs=torch.from_numpy(tgt_batchs)
    
  def getTrainBatch(self):
    """
      Return lists with shape t(str lists),l(int list) : (batch_size,senLen),(batch_size)
    """
    if(self.indic+self.batch_size<=self.dataset_size):
        t,l=self.train_batchs[self.indic:self.indic+self.batch_size],self.tgt_batchs[self.indic:self.indic+self.batch_size]
        self.indic+=self.batch_size
        return t,l,0
    else:
        t=self.train_batchs[self.indic:self.dataset_size]+self.train_batchs[0:self.batch_size-(self.dataset_size-self.indic)]
        l=self.tgt_batchs[self.indic:self.dataset_size]+self.tgt_batchs[0:self.batch_size-(self.dataset_size-self.indic)]
        self.indic=self.batch_size-(self.dataset_size-self.indic)
        return t,l,1
  def reorderForEval(self):
    whole=list(zip(self.train_batchs,self.tgt_batchs))
    whole.sort(key=lambda x:len(x[0]))
    self.train_batchs,self.tgt_batchs=zip(*whole)
    
  def getTestBatch(self):
    count=0
    minlen=maxlen=len(self.train_batchs[self.indic])
#    fobidlen=min(2*minlen,minlen+1000)
    fobidlen=minlen+1000
    example=[[self.train_batchs[self.indic]],[self.tgt_batchs[self.indic]]]
    self.indic+=1
    print(self.indic)
    batch_size= self.batch_size if minlen<=1e4 else 32
    while(True):
      if(count>=batch_size or self.indic>=self.dataset_size): break
      maxlen=max(len(self.train_batchs[self.indic]),maxlen)
      if(maxlen> fobidlen): break
      count+=1
      example[0].append(self.train_batchs[self.indic])
      example[1].append(self.tgt_batchs[self.indic])
      self.indic+=1
    example[0]=[ i+['<pad>']*(maxlen-len(i))  for i in example[0]]
    stop=False
    if(self.indic>=self.dataset_size):
      self.indic=0
      stop=True
    return example,stop
  def getPredictBatch(self):
    example=[self.train_batchs[self.indic]]
    self.indic+=1
    p=False
    if(self.indic>=self.dataset_size): p=True
    return example,p
      
  def padtoMaxLen(self,maxLen=None,DoPad=True): # pad all senten to max Len.
    """
      Repeat the source sentences until maxSenLen, note that we could also pad '<pad>' to the sentences for
      matching the maxSenLen.
    """
    if(maxLen==None):
      maxLen=self.maxSenLen
    if(DoPad):
      for i in range(0,self.dataset_size):
        length=len(self.train_batchs[i])
        if(length==maxLen): continue
        self.train_batchs[i].extend(['<pad>']*(maxLen-length))
    else:
      for i in range(0,self.dataset_size):
        length=len(self.train_batchs[i])
        if(length==maxLen): continue
        duplices=maxLen//length
        self.train_batchs[i]=self.train_batchs[i]*duplices+self.train_batchs[i][:maxLen-duplices*length]
    
  @staticmethod
  def load_devSet(x_path,y_path,wantLine=2):
    num=0
    x=open(x_path,'r')
    y=open(y_path,'r')
    x_result=[]
    y_result=[]
    x.readline()
    y.readline()
    while(True):
      x_line=x.readline()
      if(x_line==''): break
      num+=1
      x_result.append(x_line.split(',')[wantLine].split(' '))
      if(x_result[-1][-1]=='\n'): x_result[-1].pop()
      y_result.append(int(y.readline().split(',')[1])-1)
    x.close()
    y.close()
    return x_result,y_result,num
  
  
  
