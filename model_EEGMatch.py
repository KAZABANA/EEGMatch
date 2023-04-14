# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 15:29:11 2021

@author: user
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from typing import List, Dict
from sklearn import metrics

class feature_extractor(nn.Module):
    def __init__(self,hidden_1,hidden_2):
         super(feature_extractor,self).__init__()
         self.fc1=nn.Linear(310,hidden_1)
         self.fc2=nn.Linear(hidden_1,hidden_2)
         self.dropout1 = nn.Dropout(p=0.25)
         self.dropout2 = nn.Dropout(p=0.25)
    def forward(self,x):
         x=self.fc1(x)
         x=F.relu(x)
         x=self.fc2(x)
         x=F.relu(x)
         return x
    def get_parameters(self) -> List[Dict]:
         params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
                  ]
         return params  

class discriminator_DG(nn.Module):
    def __init__(self,hidden_1):
         super(discriminator_DG,self).__init__()
         self.fc1=nn.Linear(hidden_1,hidden_1)
         self.fc2=nn.Linear(hidden_1,3)
         self.dropout1 = nn.Dropout(p=0.25)
    def forward(self,x):
         x=self.fc1(x)
         x=F.relu(x)
         x=self.dropout1(x)
         x=self.fc2(x)
         return x
    def get_parameters(self) -> List[Dict]:
         params = [
            {"params": self.fc1.parameters(), "lr_mult": 1},
            {"params": self.fc2.parameters(), "lr_mult": 1},
                  ]
         return params


class Domain_adaption_model(nn.Module):
   def __init__(self,hidden_1,hidden_2,hidden_3,hidden_4,num_of_class,low_rank,max_iter,upper_threshold,lower_threshold,temp=1):
       super(Domain_adaption_model,self).__init__()
       self.fea_extrator_f= feature_extractor(hidden_1,hidden_2)
       self.fea_extrator_g= feature_extractor(hidden_3,hidden_4)
       self.U=nn.Parameter(torch.randn(low_rank,hidden_2),requires_grad=True)
       self.V=nn.Parameter(torch.randn(low_rank,hidden_4),requires_grad=True)
       self.P=torch.randn(num_of_class,hidden_4)
       self.stored_mat=torch.matmul(self.V,self.P.T)
       self.max_iter=max_iter
       self.upper_threshold=upper_threshold
       self.lower_threshold=lower_threshold
       self.threshold=upper_threshold
       self.cluster_label=np.linspace(0,num_of_class-1,num_of_class)
       self.num_of_class=num_of_class
       self.temp=temp
   def forward(self,source,target,source_label):
       feature_source_f=self.fea_extrator_f(source)
       feature_target_f=self.fea_extrator_f(target)
       feature_source_g=self.fea_extrator_f(source)
       ## Update P through some algebra computations for the convenice of broadcast
       self.P=torch.matmul(torch.inverse(torch.diag(source_label.sum(axis=0))+torch.eye(self.num_of_class).cuda()),torch.matmul(source_label.T,feature_source_g))
       self.stored_mat=torch.matmul(self.V,self.P.T)
       source_predict,target_predict=self.predict_block(feature_source_f,feature_target_f)
#       source_logit  =source_predict
       source_label_feature=torch.nn.functional.softmax(source_predict, dim=1)
       target_label_feature=torch.nn.functional.softmax(target_predict, dim=1)
       ## DAC part
       sim_matrix=self.get_cos_similarity_distance(source_label_feature)
       sim_matrix_target=self.get_cos_similarity_distance(target_label_feature)
       return source_predict,feature_source_f,feature_target_f,sim_matrix,sim_matrix_target
   def predict_block(self,feature_source_f,feature_target_f):
       source_predict=torch.matmul(torch.matmul(self.U,feature_source_f.T).T,self.stored_mat)
       target_predict=torch.matmul(torch.matmul(self.U,feature_target_f.T).T,self.stored_mat)
       return source_predict,target_predict

   def target_domain_evaluation(self,test_features,test_labels):
       self.eval()
       feature_target_f=self.fea_extrator_f(test_features)
       test_logit=torch.matmul(torch.matmul(self.U,feature_target_f.T).T,self.stored_mat.cuda())
       test_cluster=torch.nn.functional.softmax(test_logit, dim=1)
       test_cluster=np.argmax(test_cluster.cpu().detach().numpy(),axis=1)
       test_labels=np.argmax(test_labels.cpu().detach().numpy(),axis=1)
       test_predict=np.zeros_like(test_labels)
       for i in range(len(self.cluster_label)):
           cluster_index=np.where(test_cluster==i)[0]
           test_predict[cluster_index]=self.cluster_label[i]
       acc=np.sum(test_predict==test_labels)/len(test_predict)
       nmi=metrics.normalized_mutual_info_score(test_predict,test_labels)
       return acc,nmi   
   def cluster_label_update(self,source_features,source_labels):
       self.eval()
       feature_source_f=self.fea_extrator_f(source_features)
       source_logit=torch.matmul(torch.matmul(self.U,feature_source_f.T).T,self.stored_mat.cuda())
       source_cluster=np.argmax(torch.nn.functional.softmax(source_logit, dim=1).cpu().detach().numpy(),axis=1)
       source_labels=np.argmax(source_labels.cpu().detach().numpy(),axis=1)
       for i in range(len(self.cluster_label)):
           samples_in_cluster_index=np.where(source_cluster==i)[0]
           label_for_samples=source_labels[samples_in_cluster_index]
           if len(label_for_samples)==0:
              self.cluster_label[i]=0
           else:
              label_for_current_cluster=np.argmax(np.bincount(label_for_samples))
              self.cluster_label[i]=label_for_current_cluster
       source_predict=np.zeros_like(source_labels)
       for i in range(len(self.cluster_label)):
           cluster_index=np.where(source_cluster==i)[0]
           source_predict[cluster_index]=self.cluster_label[i]
       acc=np.sum(source_predict==source_labels)/len(source_predict)
       nmi=metrics.normalized_mutual_info_score(source_predict,source_labels)
       return acc,nmi
   def sharpen(self,predict,t):
       e=torch.sum((predict)**(1/t),dim=1).unsqueeze(dim=1)
       predict=(predict**(1/t))/e.expand(len(predict),self.num_of_class)
       return predict
   def predict(self,test_features,sharpen=1):
       self.eval()
       feature_target_f=self.fea_extrator_f(test_features)
       test_logit=torch.matmul(torch.matmul(self.U,feature_target_f.T).T,self.stored_mat.cuda())
       test_cluster=torch.nn.functional.softmax(test_logit, dim=1)
       if sharpen==1:
           test_cluster=self.sharpen(test_cluster,self.temp)
       test_predict=torch.zeros_like(test_cluster)
       for i in range(len(self.cluster_label)):
           temp=test_cluster[:,np.where(self.cluster_label==i)[0]]    
           test_predict[:,i]=temp.squeeze()   
       return test_predict
   def get_cos_similarity_distance(self, features):
        """Get distance in cosine similarity
        :param features: features of samples, (batch_size, num_clusters)
        :return: distance matrix between features, (batch_size, batch_size)
        """
        # (batch_size, num_clusters)
        features_norm = torch.norm(features, dim=1, keepdim=True)
        # (batch_size, num_clusters)
        features = features / features_norm
        # (batch_size, batch_size)
        cos_dist_matrix = torch.mm(features, features.transpose(0, 1))
        return cos_dist_matrix
   def get_cos_similarity_by_threshold(self, cos_dist_matrix):
        """Get similarity by threshold
        :param cos_dist_matrix: cosine distance in matrix,
        (batch_size, batch_size)
        :param threshold: threshold, scalar
        :return: distance matrix between features, (batch_size, batch_size)
        """
        device = cos_dist_matrix.device
        dtype = cos_dist_matrix.dtype
        similar = torch.tensor(1, dtype=dtype, device=device)
        dissimilar = torch.tensor(0, dtype=dtype, device=device)
        sim_matrix = torch.where(cos_dist_matrix > self.threshold, similar,
                                 dissimilar)
        return sim_matrix
   def compute_indicator(self,cos_dist_matrix):
       device = cos_dist_matrix.device
       dtype = cos_dist_matrix.dtype
       selected = torch.tensor(1, dtype=dtype, device=device)
       not_selected = torch.tensor(0, dtype=dtype, device=device)
       w2=torch.where(cos_dist_matrix < self.lower_threshold,selected,not_selected)
       w1=torch.where(cos_dist_matrix > self.upper_threshold,selected,not_selected)
       w = w1 + w2
       nb_selected=torch.sum(w)
       return w,nb_selected
   def update_threshold(self, epoch: int):
        """Update threshold
        :param threshold: scalar
        :param epoch: scalar
        :return: new_threshold: scalar
        """
        n_epochs = self.max_iter
        diff = self.upper_threshold - self.lower_threshold
        eta = diff / n_epochs
#        eta=self.diff/ n_epochs /2
        # First epoch doesn't update threshold
        if epoch != 0:
            self.upper_threshold = self.upper_threshold-eta
            self.lower_threshold = self.lower_threshold+eta
        else:
            self.upper_threshold = self.upper_threshold
            self.lower_threshold = self.lower_threshold
        self.threshold=(self.upper_threshold+self.lower_threshold)/2
#        print(">>> new threshold is {}".format(new_threshold), flush=True)
   def get_parameters(self) -> List[Dict]:
       params = [
        {"params": self.fea_extrator_f.fc1.parameters(), "lr_mult": 1},
        {"params": self.fea_extrator_f.fc2.parameters(), "lr_mult": 1},
        {"params": self.fea_extrator_g.fc1.parameters(), "lr_mult": 1},
        {"params": self.fea_extrator_g.fc2.parameters(), "lr_mult": 1},
        {"params": self.U, "lr_mult": 1},
        {"params": self.V, "lr_mult": 1},
            ]
       return params

