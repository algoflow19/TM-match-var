#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:34:04 2019

@author: jace-belere
"""

import pandas as pd
import sys

def readCorpus(output_file,input_file,lineWant):
  bad_count=0
  line_count=1
  with open(input_file,'r') as to_read:
    line=to_read.readline()
    while(True):
      line_count+=1
      line=to_read.readline()
      if(not line): break
      l=line.split(',')
      if(len(l)<2):
        bad_count+=1
        print(line+" <-- Read a bad line at:%d, and it is the %d bad line".format(line_count,bad_count))
        continue
      else:
        output_file.write(l[lineWant])
#        print(l[2])


wantWord=1
out_file=open('char_corpus.txt',"w+")
readCorpus(out_file,'../train_data.csv',wantWord)
readCorpus(out_file,'../test_data.csv',wantWord)
out_file.close()
