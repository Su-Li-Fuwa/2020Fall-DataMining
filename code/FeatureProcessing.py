from sklearn.decomposition import PCA
import torch
import lightgbm as lgb
import numpy as np
import copy
from util import softmax

usePCA = True
useLight = True
useEncoding = True
PCAthreshold = 0.3

def featureSelection(data, indices, timer):
    x, y, edge_index, edge_weight = data.x, data.y, data.edge_index.transpose(0, 1), data.edge_weight
    y = y.cpu().numpy()
    if x.shape[1] == 1:
        # No feature, only use edge data
        hasfeature = False
        x=np.zeros((x.shape[0], 1))
    else:
        hasfeature = True
        x = x.drop('node_index', axis=1).to_numpy()

        #keepNum = 50 #int(x.shape[1]/10)
        pca=PCA(n_components = PCAthreshold, svd_solver = 'full')
        #pca=PCA(n_components = keepNum, svd_solver = 'full')
        print("Before PCA, input shape:", x.shape)
        pca_x=pca.fit_transform(x)
        print("After PCA, input shape:", pca_x.shape)

        if useLight:
            #print(indices)
            totalSet = lgb.Dataset(x[indices[:int(0.8*len(indices))]], y[indices[:int(0.8*len(indices))]])
            validSet = lgb.Dataset(x[indices[int(0.8*len(indices))+1:]], y[indices[int(0.8*len(indices))+1:]])
            params = {
                'boosting_type': 'gbdt',
                #'verbose': -1,
                'random_state': 0,
                'silent': -1,
                'subsample': 0.9,
                'subsample_freq': 1,
                "num_leaves":63,  
                "min_data_in_leaf":1,
                'n_jobs': 10,
                'objective': 'multiclass',
                'num_class': max(y)+1
            }
            model = lgb.train(params,totalSet, num_boost_round=1, valid_sets=validSet)
            impTable = list(model.feature_importance())
            tmpList = copy.deepcopy(impTable)
            tmpList.sort(reverse=True)
            threshold = tmpList[0]
            tmpRecord = threshold
            for i in range(len(tmpList)):
                if tmpList[i] != tmpRecord:    
                    threshold = tmpRecord
                    tmpRecord = tmpList[i]
                if i>len(pca_x):    break
            sigList = []
            print(threshold)
            for i in range(len(impTable)):
                if impTable[i] > threshold:
                    sigList.append(i)
            print(sigList)
            print(pca_x.shape)
            x = x[:,sigList]
            x = np.hstack((x, pca_x))
        else:
            x = pca_x
        
    print("After selection:", x.shape)
    if useEncoding:#not hasfeature or x.shape[1] < 2*(max(y)+1):
        idx = np.array(list(range(len(edge_index))))
        np.random.shuffle(idx)

        edge_append = np.ones((x.shape[0],max(y)+1))
        #print(edge_append.shape, len(edge_index), edge_weight.shape)
        timer.record()
        for i in idx[:1000]:
            if y[edge_index[i][0]] != -1:
                edge_append[edge_index[i][1]][y[edge_index[i][0]]] += edge_weight[i]
            if y[edge_index[i][1]] != -1:
                edge_append[edge_index[i][0]][y[edge_index[i][1]]] += edge_weight[i]
        estimateN = min(timer.record(phase = 0.1)+1000, len(edge_index))
        print("****** estimateN ****** len ****", estimateN, len(edge_index))
        for i in idx[1000:estimateN]:
            if y[edge_index[i][0]] != -1:
                edge_append[edge_index[i][1]][y[edge_index[i][0]]] += edge_weight[i]
            if y[edge_index[i][1]] != -1:
                edge_append[edge_index[i][0]][y[edge_index[i][1]]] += edge_weight[i]
            if (timer.record()):    break
        edge_append = softmax(edge_append)
        edgeLoss = torch.nn.functional.cross_entropy(torch.from_numpy(edge_append[indices]), torch.from_numpy(y[indices]), size_average=True)
        if (edgeLoss < 1.5):
            if not hasfeature: x = edge_append
            else:   x = np.hstack((x, edge_append))
        
    print("input shape:", x.shape)
    x = torch.tensor(x, dtype=torch.float)
    data.x = x
    return data