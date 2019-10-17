#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 08:56:27 2019

@author: jace-belere
"""

# Original dataset size: 199961
import random


train_path='../train_data.csv'
tlabel_path='../train_labels.csv'
new_train_path='../new_train_data.csv'
new_tlabes_path='../new_train_labels.csv'
dev_path='../dev_data.csv'
dlabel_path='../dev_labels.csv'


dev_rate=0.1 # 9 vs 1 For train:test
is_first_row_title=True
train_set=open(train_path,'r')
train_labels=open(tlabel_path,'r')

dev_set=open(dev_path,'w+')
dev_labels=open(dlabel_path,'w+')
new_train=open(new_train_path,'w+')
new_tlabels=open(new_tlabes_path,'w+')

first_line=train_set.readline()
dev_set.write(first_line)
new_train.write(first_line)
first_line=train_labels.readline()
dev_labels.write(first_line)
new_tlabels.write(first_line)

for line in train_set:
  if(random.random()<=dev_rate):
    dev_set.write(line)
    dev_labels.write(train_labels.readline())
  else:
    new_train.write(line)
    new_tlabels.write(train_labels.readline())
    
train_set.close()
train_labels.close()
dev_set.close()
dev_labels.close()
new_train.close()
new_tlabels.close()



