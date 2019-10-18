#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:19:53 2019

@author: jace-belere
"""
from utils import DataSet
from model_embedding import ModelEmbeddings
import argparse
from textcnn import textCNN
import torch
import numpy as np

def main():
  parser = argparse.ArgumentParser(description='Text Classification for tinyMind task.')
  parser.add_argument('--mode', type=str, required=True, help='choose a mode: [train,test]')
  parser.add_argument('--train_data', default='../new_train_data.csv', type=str)
  parser.add_argument('--train_labels', default='../new_train_labels.csv', type=str)
  parser.add_argument('--dev_data', default='../dev_data.csv', type=str)
  parser.add_argument('--dev_labels', default='../dev_labels.csv', type=str)
  parser.add_argument('--predict_data', default='../dav_data.csv', type=str)
  
  parser.add_argument('--pretrain_vector',default='../totVector.txt',type=str)
  parser.add_argument('--max_epoch',default=8,type=int)
  parser.add_argument('--lr',default=0.0008,type=float)
  parser.add_argument('--device',default='cpu',type=str,help='Please assign it with cpu or cuda:0')
  parser.add_argument('--batch_size',default=128,type=int)
  parser.add_argument('--embedding_size',default=200,type=int)
  parser.add_argument('--cls_num',default=27,type=int)
  parser.add_argument('--l1_channels_num',default=100,type=int)
  parser.add_argument('--model_save_path',default='../save_model/model.bin',type=str)
  parser.add_argument('--embedding_save_path',default='../save_model/embed.bin',type=str)
  parser.add_argument('--predict_writeto',default='../predict_reslt.txt',type=str)
#  parser.add_argument('--cls_num',default=27,type=int) 
  args = parser.parse_args()
  
  if(args.mode=='train'):
    train(args)
  elif(args.mode=='predict'):
    predict(args)
  else:
    print('You must choose if you want to train or test the model.')
  return 0

def train(args):
  device=torch.device(args.device)
  cls_embed=ModelEmbeddings.load_from_file(args.pretrain_vector)
  train_data=DataSet(args.train_data,args.train_labels,args.batch_size)
  train_data.padtoMaxLen()
  print('loading dev set...')
#  devSet,devlabels,devNum=DataSet.load_devSet(args.dev_data,args.dev_labels)
  dev_dataset=DataSet(args.dev_data,args.dev_labels,args.batch_size)
  dev_dataset.reorderForEval()
  print('Done dev loading.')
  model=textCNN(args.embedding_size,args.cls_num,args.l1_channels_num)
  model.train()
  model = model.to(device)
  cls_embed=cls_embed.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
  Loss_fun=torch.nn.CrossEntropyLoss()
  
  print('begin Maximum Likelihood training')
  epoch=0
  sum_loss=0
  step=0
  while(True):
    optimizer.zero_grad()
    t,l,p=train_data.getTrainBatch()
    l=torch.tensor(l,device=device)
#    print("Doing word embed")
#    print(type(t))
#    print(type(l))
#    print(l.requires_grad)
    
    t=cls_embed(t,device=device)
#    t=t.detach()
#    print("Done word embed")
#    print("feed into model...")
    
    outPuts=model(t)
#    print("Get answer!")
#    print("Caluating Loss")
#    print(outPuts.size())
#    print(l.size())
#    print(l)
#    print(outPuts)
    loss=Loss_fun(outPuts,l)
    sum_loss+=loss
    print("epoch:{0},step:{1},train loss:{2}".format(epoch,step,loss))
    step+=1
#    print("Doing backPro")
    loss.backward()
#    print("Done backPro")
#    print("Setping...")
    optimizer.step()
#    print("Done Step!")
#    print('Current Batch Loss:{0}'.format(loss))
    if(p):
      epoch+=p
      print("Epoch mean Loss:{0}".format(sum_loss/step))
      step=0
      sum_loss=0
      accuary,F1=test(args,model,dev_dataset,cls_embed,args.cls_num,device)
      if(model.lastScore<F1):
        print("F1 score grow from {0} to {1}, save model...".format(model.lastScore,F1))
        model.lastScore=F1
        torch.save(model.state_dict(),args.model_save_path)
        torch.save(cls_embed,args.embedding_save_path)
    if(epoch==args.max_epoch): break
  
def calcuate_scores(outputs,labels,cls_num):
  cls_tTum=np.array([1e-6]*cls_num) # total hit on class.
  cls_tNum=np.array([1e-6]*cls_num) # total pre cls num.
  cls_tAum=np.array([1e-6]*cls_num) # total accruary hit.
#  print(type(outputs))
#  print(type(labels))
  assert(len(outputs)==len(labels))
  for i in range(len(outputs)):
#    print("Model predict: {0},Labels: {1}".format(outputs[i],labels[i]))
    cls_tTum[outputs[i]]+=1
    cls_tNum[labels[i]]+=1
    if(outputs[i]==labels[i]):
      cls_tAum[outputs[i]]+=1
  P=cls_tAum/cls_tTum
  R=cls_tAum/cls_tNum
#  print(P)
#  print(R)
  return P.mean(),((2*P*R)/(P+R)).mean()

def test(args,model,dataSet,cls_embed,cls_num,device):
  """
    Do Eval set analysis
  """
  print('Start dev set eval...')
  model.eval()
  result_list=[]
  label_list=[]
 # print(dev_data)
  while(True):
 #   print(example)
    example,stop=dataSet.getTestBatch()
    to_in=cls_embed(example[0],device)
    to_in=model(to_in)
    result_list.extend((torch.argmax(to_in,-1).cpu()).tolist())
    label_list.extend(example[1])
    if(stop): break
  accuary,f1=calcuate_scores(result_list,label_list,cls_num)
  print("Get Average Accuary:{0} ,F1 score:{1}".format(accuary,f1))
  model.train()
  return accuary,f1

def predict(args):
  device=torch.device(args.device)
  cls_embed=torch.load(args.embedding_save_path)
  model=textCNN(args.embedding_size,args.cls_num,args.l1_channels_num)
  model.load_state_dict(args.model_save_path)
  model.eval()
  model=model.to(device)
  cls_embed.to(device)
  dev_dataset=DataSet(args.predict_data,None,args.batch_size)
  towrite=open(args.predict_writeto,"w+")
  towrite.write("idx,labels\n")
  idx=0
  print("Begin Predict task...")
  while(True):
    example,p=dev_dataset.getPredictBatch()
    example=cls_embed(example)
    out=model(example)
    out=torch.argmax(-1).item()+1
    towrite.write("{0},{1}\n".format(idx,out))
    idx+=1
    if(p): break
  towrite.close()
  print("Predict task Done!")
    
    
#embed_size=5
#cls_num=3
#l1_channels=4
#model=textCNN(embed_size,cls_num,l1_channels)
#outputs=[1,2,3,4,4,3,1,4]
#labels=[1,2,3,4,1,2,3,4]
#calcuate_scores(outputs,labels,7)

if __name__ == '__main__':
    main()
