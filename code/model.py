import numpy as np
import pandas as pd
import copy

import torch
from torch_geometric.data import Data

# New packet import
import FeatureProcessing as fp
import util as u
from network import *
from ModelSelection import *

import random
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
fix_seed(1234)

class Model:

    def __init__(self):
        self.device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

    def generate_pyg_data(self, data, timer):
        print("############## DATA PROCESSING START ##############")
        x = data['fea_table']
        num_nodes = x.shape[0]

        train_indices = data['train_indices']
        test_indices = data['test_indices']

        df = data['edge_file']
        edge_index = df[['src_idx', 'dst_idx']].to_numpy()
        edge_index = sorted(edge_index, key=lambda d: d[0])
        edge_weight = df['edge_weight'].to_numpy()

        y = torch.zeros(num_nodes, dtype=torch.long)
        y = y - 1 
        inds = data['train_label'][['node_index']].to_numpy()
        train_y = data['train_label'][['label']].to_numpy()
        y[inds] = torch.tensor(train_y, dtype=torch.long)

        edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
        edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
        
        # feature selection
        data = fp.featureSelection(Data(x=x, edge_index=edge_index, y=y, edge_weight=edge_weight),\
                                   train_indices, timer)
        num_nodes = data.x.size(0)
        data.num_nodes = num_nodes

        # validation set
        np.random.shuffle(train_indices)
        lenTrain = int(len(train_indices)*0.7)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_indices[:lenTrain]] = 1
        data.train_mask = train_mask

        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask[train_indices[lenTrain+1:]] = 1
        data.val_mask = val_mask

        total_mask = torch.zeros(num_nodes, dtype=torch.bool)
        total_mask[train_indices] = 1
        data.total_mask = total_mask

        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[test_indices] = 1
        data.test_mask = test_mask

        print("############## DATA PROCESSING END ##############")
        return data

    def val_acc(self, model, data):
        model.eval()
        data = data.to(self.device)
        with torch.no_grad():
            predtv = model(data)[data.val_mask].max(1)[1]
        acctv = (predtv == data.y[data.val_mask]).tolist()
        return sum(acctv)/len(acctv)

    def train_predict(self, data, time_budget, n_class,schema):
        timer = u.Timer(time_budget)

        timer.record(phase = 0)
        data = self.generate_pyg_data(data, timer)
        data = data.to(self.device)

        ######## Simple AutoML Part ########
        
        '''
            model selection
        '''
        for i in range(len(defaultArgsList)):
            defaultArgsList[i]['features_num'] = data.x.size()[1]
            defaultArgsList[i]['num_class'] = int(max(data.y)) + 1
            defaultArgsList[i]['data'] = data
            defaultArgsList[i]['device'] = self.device
        torch.cuda.empty_cache()
        selectedModel, selectedPara = modelSelection(timer)
        # search dropout ratio 0.2~0.8 from small to large
        # torch.cuda.empty_cache()
        
        ###################################
        timer.record(phase = 2)
        print(selectedModel, ' ', selectedPara)
        ### Bagging is not good
        firstT = 1
        while True:
            model = selectedModel(selectedPara).to(self.device)
            #model = GAT(defaultArgsList[0]).to(self.device)
            _ = model.myTrain(False, timer.record())
            if not firstT and _<0:  break
            tmp = model.pred().cpu().numpy()
            print("Selected loss: %f\t"%(_))
            if firstT:
                #predict = torch.exp(torch.tensor(-_))*torch.exp(tmp).detach()
                predict = np.exp(-_)*np.exp(tmp)
                firstT = 0
            else:
                predict += np.exp(-_)*np.exp(tmp)
            if _<0: break

        ######## Simple AutoML Part ########
        return torch.from_numpy(predict).max(1)[1].cpu().numpy().flatten()
