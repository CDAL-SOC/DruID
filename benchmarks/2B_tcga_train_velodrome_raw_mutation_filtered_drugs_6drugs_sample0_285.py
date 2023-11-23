#!/usr/bin/env python
# coding: utf-8

# This notebook uses the Velodrome architecture on the Druid dataset, for baseline comparison. The input data is the mutation profile, from the Foundation One report.
# 
# Velodrome reference: https://www.nature.com/articles/s42256-021-00408-w

# Here, models are created, one for each drug and aggregated to get the results.

# #### Imports

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import sys

# sys.path.append("../../vae_zinb_reprn/")
sys.path.append("../src/")


# In[3]:


import datetime
import logging
import os
import time
import torch
import random
import pickle


# In[4]:


from torch import nn
from torch.nn import functional as F

from functools import cached_property

from torch.nn import Linear, ReLU, Sequential
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score
from sklearn.model_selection import train_test_split


from datasets_drug_filtered import (
    AggCategoricalAnnotatedCellLineDatasetFilteredByDrug,
    AggCategoricalAnnotatedTcgaDatasetFilteredByDrug,
)

from utils import get_kld_loss, get_zinb_loss


# In[5]:



# In[6]:

# Uses FuncVelov3.py from https://github.com/hosseinshn/Velodrome
from FuncVelov3 import *


# In[7]:

sample_id = 0
# arguments class for hyperparameters like weight decay, learning rate etc
class arguments():
    def __init__(self):
        self.seed = 42
        self.ldr = 0.5 # dropout
        self.hd = 1 # for network architecture
        self.bs = 64 
        self.wd = 0.5 # weight decay
        self.wd1 = 0.1
        self.wd2 = 0.1
        self.lr = 0.001 # learning rate
        self.lr1 = 0.005
        self.lr2 = 0.005
        self.lam1 = 0.005
        self.lam2 = 0.005
        self.epoch = 30
        self.gpu = 0
        self.save_logs = f"../data/model_checkpoints/tcga_velodrome_logs_raw_mutations_285_filtered_drugs_6_sample{sample_id}/"
        self.save_models = f"../data/model_checkpoints/tcga_velodrome_models_raw_mutations_285_filtered_drugs_6_sample{sample_id}/"
        self.save_results = f"../data/model_checkpoints/tcga_velodrome_results_raw_mutations_285_filtered_drugs_6_sample{sample_id}/"
        os.makedirs(self.save_logs, exist_ok=True)
        os.makedirs(self.save_models, exist_ok=True)
        os.makedirs(self.save_results, exist_ok=True)
args = arguments()


# In[8]:


torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# To avoid randomness in DataLoaders - https://pytorch.org/docs/stable/notes/randomness.html
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
g = torch.Generator()
g.manual_seed(0)

# In[9]:


torch.multiprocessing.set_sharing_strategy('file_system')


# #### Model Definition

# In[10]:


# Velodrome definition from https://github.com/hosseinshn/Velodrome
def VelodromeNetwork(args, X):
    IE_dim = X.shape[1]

    class Net1(nn.Module):
        def __init__(self, args):
            super(Net1, self).__init__()

            self.features = nn.Sequential(
                nn.Linear(IE_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(args.ldr),
                nn.Linear(512, 128),
                nn.Sigmoid()) 

        def forward(self, x):
            out = self.features(x)
            return out     
    
    class Net2(nn.Module):
        def __init__(self, args):
            super(Net2, self).__init__()

            self.features = torch.nn.Sequential(
                nn.Linear(IE_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(args.ldr),
                nn.Linear(256, 256),
                nn.Sigmoid()
                ) 

        def forward(self, x):
            out = self.features(x)
            return out            

    class Net3(nn.Module):
        def __init__(self, args):
            super(Net3, self).__init__()    

            self.features = torch.nn.Sequential(
                nn.Linear(IE_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(args.ldr),
                nn.Linear(128, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(args.ldr),
                nn.Linear(128, 128),
                nn.Sigmoid()
                )                        

        def forward(self, x):
            out = self.features(x)
            return out

    class Net4(nn.Module):
        def __init__(self, args):
            super(Net4, self).__init__()    

            self.features = torch.nn.Sequential(
                nn.Linear(IE_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(args.ldr),
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(args.ldr),
                nn.Linear(64, 64),
                nn.Sigmoid()
                )                        

        def forward(self, x):
            out = self.features(x)
            return out        
        
    if args.hd == 1:
        Model = Net1(args)
    elif args.hd == 2:
        Model = Net2(args)
    elif args.hd == 3:
        Model = Net3(args)
    elif args.hd == 4:
        Model = Net4(args)        

    class Pred(nn.Module):
        def __init__(self, args):
            super(Pred, self).__init__()
            if args.hd == 1: 
                dim = 128            
            if args.hd == 2: 
                dim = 256
            if args.hd == 3:
                dim = 128
            if args.hd == 4:
                dim = 64                
            self.pred = torch.nn.Sequential(
                nn.Linear(dim, 1)) 

        def forward(self, x):
            out = self.pred(x)
            return out     
    torch.manual_seed(args.seed)    
    Predict_1 = Pred(args)
    torch.manual_seed(args.seed*2)
    Predict_2 = Pred(args)
    torch.manual_seed(args.seed)
        
    return Model, Predict_1, Predict_2


# In[11]:


# Class definition for running test bed evaluation
# start with gene exp, one drug per model, 385 instances then average out for test bed eval
# or 385 output variables
# Here we use one network for one drug to test
class VelodromeOneDrug(nn.Module):
    def __init__(self, drug_name, args, trainLoader_1, trainLoader_2, trainULoader,
                 TX_val, Ty_val, X1_train, w1=None, w2=None):
        super(VelodromeOneDrug, self).__init__()

        self.trainLoader_1 = trainLoader_1
        self.trainLoader_2 = trainLoader_2    

        self.trainULoader = trainULoader
        self.TX_val = TX_val
        self.Ty_val = Ty_val
        self.X1_train = X1_train
        self.drug_name = drug_name.replace("/", "_")
        self.model, self.pred1, self.pred2 = VelodromeNetwork(args=args, X=self.X1_train)
        self.w1 = w1
        self.w2 = w2
        
    def train(self, args=arguments()):
        # init model        
        opt = torch.optim.Adagrad(self.model.parameters(), lr=args.lr, weight_decay = args.wd)
        opt1 = torch.optim.Adagrad(self.pred1.parameters(), lr=args.lr1, weight_decay = args.wd1)
        opt2 = torch.optim.Adagrad(self.pred2.parameters(), lr=args.lr2, weight_decay = args.wd2)
    
        loss_fun = torch.nn.MSELoss()
        total_val = []
        total_aac = []

        train_loss = []
        consistency_loss = []
        covariance_loss = []
        train_pr1 = []
        train_pr2 = []
        val_loss = []
        val_pr = []
    
        train_pred = []
        w1 = []
        w2 = []
        
        best_pr = 0
        torch.save(self.model.state_dict(), os.path.join(args.save_models, f'{self.drug_name}Best_Model.pt'))
        torch.save(self.pred1.state_dict(), os.path.join(args.save_models, f'{self.drug_name}Best_Pred1.pt'))
        torch.save(self.pred2.state_dict(), os.path.join(args.save_models, f'{self.drug_name}Best_Pred2.pt')) 
        
        
        for ite in range(args.epoch):
            torch.autograd.set_detect_anomaly(True)
            pred_loss, coral_loss, con_loss, epoch_pr1, epoch_pr2, loss1, loss2 = train(args, self.model, self.pred1, self.pred2, loss_fun, opt, opt1, opt2, self.trainLoader_1, self.trainLoader_2, self.trainULoader)

            train_loss.append(pred_loss + coral_loss + con_loss)      
            train_loss.append(pred_loss + con_loss)      
            consistency_loss.append(con_loss)
            covariance_loss.append(coral_loss)
            train_pr1.append(epoch_pr1)
            train_pr2.append(epoch_pr2)

            w1.append(loss1)
            w2.append(loss2)

            epoch_val_loss, epoch_Val_pr,_ = validate_workflow(args, self.model, self.pred1, self.pred2, [torch.mean(torch.stack(w1)), torch.mean(torch.stack(w2))], loss_fun, self.TX_val, self.Ty_val)
            val_loss.append(epoch_val_loss)
            val_pr.append(epoch_Val_pr)                      

            if epoch_Val_pr > best_pr: 
                best_pr = epoch_Val_pr
                torch.save(self.model.state_dict(), os.path.join(args.save_models, f'{self.drug_name}Best_Model.pt'))
                torch.save(self.pred1.state_dict(), os.path.join(args.save_models, f'{self.drug_name}Best_Pred1.pt'))
                torch.save(self.pred2.state_dict(), os.path.join(args.save_models, f'{self.drug_name}Best_Pred2.pt'))

#         plots(args, train_loss, consistency_loss, covariance_loss, train_pr1, train_pr2, val_loss, val_pr)
        self.model.load_state_dict(torch.load(os.path.join(args.save_models, f'{self.drug_name}Best_Model.pt')))
        self.pred1.load_state_dict(torch.load(os.path.join(args.save_models, f'{self.drug_name}Best_Pred1.pt')))
        self.pred2.load_state_dict(torch.load(os.path.join(args.save_models, f'{self.drug_name}Best_Pred2.pt')))

        self.model.eval()
        self.pred1.eval()
        self.pred2.eval()

        _,_, preds= validate_workflow(args, self.model, self.pred1, self.pred2, [torch.mean(torch.stack(w1)), torch.mean(torch.stack(w2))], loss_fun, self.TX_val, self.Ty_val)
        total_val.append(preds.detach().numpy().flatten())
        total_aac.append(self.Ty_val.detach().numpy())
        self.w1 = w1
        self.w2 = w2

        self.model.load_state_dict(torch.load(os.path.join(args.save_models, f'{self.drug_name}Best_Model.pt')))
        self.pred1.load_state_dict(torch.load(os.path.join(args.save_models, f'{self.drug_name}Best_Pred1.pt')))
        self.pred2.load_state_dict(torch.load(os.path.join(args.save_models, f'{self.drug_name}Best_Pred2.pt')))


        final_pred = list(itertools.chain.from_iterable(total_val))
        final_labels = list(itertools.chain.from_iterable(total_aac))
        return self.w1, self.w2
                                   
    def forward(self, X):
        self.model.load_state_dict(torch.load(os.path.join(args.save_models, f'{self.drug_name}Best_Model.pt')))
        self.pred1.load_state_dict(torch.load(os.path.join(args.save_models, f'{self.drug_name}Best_Pred1.pt')))
        self.pred2.load_state_dict(torch.load(os.path.join(args.save_models, f'{self.drug_name}Best_Pred2.pt')))
        self.model.eval()
        self.pred1.eval()
        self.pred2.eval()
        ws = [torch.mean(torch.stack(self.w1)), torch.mean(torch.stack(self.w2))]
        w_n = torch.nn.functional.softmax(torch.stack(ws), dim=None)
        w1 = w_n[0]
        w2 = w_n[1]
        
        TX_val = torch.tensor(
            X,
            dtype=torch.float,
        )

        fx_val = self.model(TX_val)
        pred_1 = self.pred1(fx_val)
        pred_2 = self.pred2(fx_val)
        pred_val = w1*pred_1+w2*pred_2
        return pred_val
        


# In[12]:


# Class definition for running test bed evaluation
# Here we use one network for one drug to test
class TrainVelodrome():
    def __init__(self, args=args):
        super(TrainVelodrome, self).__init__()
        # load train data
        cl_dataset_train = AggCategoricalAnnotatedCellLineDatasetFilteredByDrug(
            is_train=True,
            filter_for="tcga",
            sample_id=sample_id
        )
        
        _, drug_names, _ = list(
            cl_dataset_train[: len(cl_dataset_train)].values()
        )
        unique_drugs = [
	    "CISPLATIN","PACLITAXEL","5-FLUOROURACIL","CYCLOPHOSPHAMIDE","DOCETAXEL","GEMCITABINE",
        ]
        print(len(unique_drugs))
        print(unique_drugs)
        
        
        # load test data
        cl_dataset_test = AggCategoricalAnnotatedCellLineDatasetFilteredByDrug(
            is_train=False,
            filter_for="tcga",
            sample_id=sample_id
        )
        # unlabelled TCGA dataset
        tcga_train = AggCategoricalAnnotatedTcgaDatasetFilteredByDrug(
            is_train=True,
            filter_for="tcga",
            sample_id=sample_id
        )
        
        tcga_train_drug_specific_df = tcga_train.tcga_response # considering all train TCGA entities
        tcga_train_mut_df = tcga_train.raw_mutations_285_genes[tcga_train.raw_mutations_285_genes.index.isin(tcga_train_drug_specific_df.submitter_id)]
        
        tcga_train_merged_df = pd.merge(tcga_train_mut_df.reset_index(), tcga_train_drug_specific_df.drop("drug_name", axis = 1), on = "submitter_id").set_index("submitter_id")
        
        #X_U = tcga_train_merged_df.drop("response", axis = 1).values
        X_U = tcga_train_mut_df.values
        
        trainUDataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_U))
        self.trainULoader = torch.utils.data.DataLoader(dataset = trainUDataset, batch_size=args.bs, shuffle=True, num_workers=1, drop_last=True, generator=g, worker_init_fn=seed_worker)
        
        self.velodrome_models_per_drug = {}
        for drug_name in unique_drugs:
            # filter out for the specific drug
            train_drug_specific_df = cl_dataset_train.y_df[cl_dataset_train.y_df["drug_name"] == drug_name]# has depmap_id, drug_name, auc 
            train_mut_df = cl_dataset_train.raw_mutations_285_genes[cl_dataset_train.raw_mutations_285_genes.index.isin(train_drug_specific_df.depmap_id)]

            train_merged_df = pd.merge(train_mut_df.reset_index(), train_drug_specific_df.drop("drug_name", axis = 1), on = "depmap_id").set_index("depmap_id")
            # convert to pytorch objects after dividing into 2 CL datasets 
            X1_train = train_merged_df.iloc[0: len(train_merged_df)//2].drop("auc", axis = 1).values
            y1_train = train_merged_df.iloc[0: len(train_merged_df)//2]["auc"].values
            X2_train = train_merged_df.iloc[len(train_merged_df)//2:].drop("auc", axis = 1).values
            y2_train = train_merged_df.iloc[len(train_merged_df)//2:]["auc"].values

            # filter out for the specific drug
            test_drug_specific_df = cl_dataset_test.y_df[cl_dataset_test.y_df["drug_name"] == drug_name]# has depmap_id, drug_name, auc 
            test_mut_df = cl_dataset_test.raw_mutations_285_genes[cl_dataset_test.raw_mutations_285_genes.index.isin(test_drug_specific_df.depmap_id)]

            test_merged_df = pd.merge(test_mut_df.reset_index(), test_drug_specific_df.drop("drug_name", axis = 1), on = "depmap_id").set_index("depmap_id")
            # convert to pytorch objects after dividing into 2 CL datasets 
            X1_test = test_merged_df.iloc[0: len(test_merged_df)//2].drop("auc", axis = 1).values
            y1_test = test_merged_df.iloc[0: len(test_merged_df)//2]["auc"].values
            X2_test = test_merged_df.iloc[len(test_merged_df)//2:].drop("auc", axis = 1).values
            y2_test = test_merged_df.iloc[len(test_merged_df)//2:]["auc"].values

            train1Dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X1_train), torch.FloatTensor(y1_train))
            self.trainLoader_1 = torch.utils.data.DataLoader(dataset = train1Dataset, batch_size=args.bs, shuffle=True, num_workers=1, drop_last=True, generator=g, worker_init_fn=seed_worker)

            train2Dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X2_train), torch.FloatTensor(y2_train))
            self.trainLoader_2 = torch.utils.data.DataLoader(dataset = train2Dataset, batch_size=args.bs, shuffle=True, num_workers=1, drop_last=True, generator=g, worker_init_fn=seed_worker)
            X_val = np.concatenate((X1_test, X2_test), axis=0)
            y_val = np.concatenate((y1_test, y2_test), axis=0)
            self.TX_val = torch.FloatTensor(X_val)
            self.Ty_val = torch.FloatTensor(y_val)
            self.X1_train = X1_train
            self.velodrome_models_per_drug[drug_name] = {"model_path": args.save_models, "w1": None, "w2": None}
#             self.velodrome_models_per_drug[drug_name] = VelodromeOneDrug(drug_name=drug_name, args=args, trainLoader_1=self.trainLoader_1, 
#                                                                  trainLoader_2=self.trainLoader_2, 
#                                                                  trainULoader=self.trainULoader,
#                                                                  TX_val=self.TX_val, Ty_val=self.Ty_val,
#                                                                  X1_train=self.X1_train)
        
        self.cl_drugs = unique_drugs
        
    def train_model(self, args=arguments()):
        # train each of the 385 models
        for k, v in self.velodrome_models_per_drug.items():
            print(f"Training model for {k}")
            velodrome_one_drug_model = VelodromeOneDrug(drug_name=k, args=args, trainLoader_1=self.trainLoader_1, 
                                                                 trainLoader_2=self.trainLoader_2, 
                                                                 trainULoader=self.trainULoader,
                                                                 TX_val=self.TX_val, Ty_val=self.Ty_val,
                                                                 X1_train=self.X1_train)
            w1, w2 = velodrome_one_drug_model.train()
            self.velodrome_models_per_drug[k]["w1"] = w1
            self.velodrome_models_per_drug[k]["w2"] = w2
            del velodrome_one_drug_model
        print("Saving the weight dict")
        with open(f'{args.save_models}/weights.pickle', 'wb') as handle:
            pickle.dump(self.velodrome_models_per_drug, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
                                   

# In[13]:


m = TrainVelodrome(args)


# In[ ]:


m.train_model()


