import torch
import torch.nn.functional as F
import time
from torch.nn import Linear
from torch_geometric.nn import GCNConv, JumpingKnowledge, GATConv
from torch.nn import BatchNorm1d

class BlankModel(torch.nn.Module):
    def __init__(self, args):
        super(BlankModel, self).__init__()
        self.lr = args['lr']
        self.data = args['data']
        self.weight_decay = args['weight_decay']
        self.beta = args['beta']
        self.device = args['device']
        self.exInterval = args['exInterval']

    def resetMetaData(self, args):
        self.lr = args['lr']
        self.weight_decay = args['weight_decay']
        self.beta = args['beta']
        self.exInterval = args['exInterval']

    def myTrain(self, isVal, breakTime = 0):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if isVal:   trainMask = self.data.train_mask
        else:       trainMask = self.data.total_mask
        valMask = self.data.val_mask
        min_loss = float('inf')
        forward = self.forward()
        lossVal = float(F.nll_loss(forward[valMask], self.data.y[valMask]))
        oldLoss = lossVal
        if breakTime: sTime = time.time()
        for epoch in range(1, 2000):
            self.train()
            optimizer.zero_grad()
            forward = self.forward()
            loss = F.nll_loss(forward[trainMask], self.data.y[trainMask])
            loss.backward()
            lossVal = (1-self.beta)*lossVal + self.beta*float(F.nll_loss(forward[valMask], self.data.y[valMask]))
            if breakTime:
                if (time.time() - sTime) > breakTime:   return -1
            # early termination
            if epoch % self.exInterval == 0:
                if (abs(lossVal - oldLoss)/lossVal < 0.05):
                    break
                oldLoss = lossVal
            optimizer.step()
        tLoss = float(F.nll_loss(forward[trainMask], self.data.y[trainMask]))
        if isVal:   return 0.7*lossVal+0.3*tLoss
        else: return tLoss
    
    def pred(self):
        self.eval()
        with torch.no_grad():
            pred = self.forward()[self.data.test_mask]
        return pred

    #@abstractmethod
    def forward(self):
        pass

class GAT(BlankModel):

    def __init__(self, args):
        self.args = args
        super(GAT, self).__init__(args)
        self.conv1 = GATConv(args['hidden'], args['hidden'], args['headslayers'])
        self.conv2 = GATConv(args['hidden']*args['headslayers'], args['hidden'])
        self.outdim = sum([args['hidden'], args['hidden'], args['hidden']*args['headslayers']])
        self.lin2 = Linear(self.outdim, args['num_class'])
        self.first_lin = Linear(args['features_num'], args['hidden'])
        self.bn1 = BatchNorm1d(args['hidden'])
        self.bn2 = BatchNorm1d(args['hidden']*args['headslayers'])
        self.bn3 = BatchNorm1d(args['hidden'])
        self.dropP = args['dropout_ratio']

    def resetMetaData(self, args):
        super(GAT, self).resetMetaData(args)
        self.dropP = args['dropout_ratio']
        self.reset_parameters()

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self):
        x, edge_index, edge_weight = self.data.x, self.data.edge_index, self.data.edge_weight
        x = F.leaky_relu(self.first_lin(x))
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropP, training=self.training)
        xTmp = [x]
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropP, training=self.training)
        xTmp.append(x)
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = self.bn3(x)
        x = F.dropout(x, p=self.dropP, training=self.training)
        xTmp.append(x)
        x = torch.cat(xTmp, dim=1)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

class GCN(BlankModel):

    def __init__(self, args):
        self.args = args
        # hidden = hidden*16
        super(GCN, self).__init__(args)
        self.conv1 = GCNConv(args['features_num'], args['hidden'])
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(args['headslayers'] - 1):
            self.convs.append(GCNConv(args['hidden'], args['hidden']))
            self.bns.append(BatchNorm1d(args['hidden']))
        self.lin2 = Linear(args['hidden'], args['num_class'])
        self.first_lin = Linear(args['features_num'], args['hidden'])
        self.bn1 = BatchNorm1d(args['hidden'])
        self.dropP = args['dropout_ratio']
    
    def resetMetaData(self, args):
        super(GCN, self).resetMetaData(args)
        self.dropP = args['dropout_ratio']
        self.reset_parameters()

    def reset_parameters(self):
        self.first_lin.reset_parameters()
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self):
        x, edge_index, edge_weight = self.data.x, self.data.edge_index, self.data.edge_weight
        x = F.relu(self.first_lin(x))
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropP, training=self.training)
        i = 0
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight=edge_weight))
            x = self.bns[i](x)
            i += 1
            x = F.dropout(x, p=self.dropP, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

netList = [GAT, GCN]
GAT_args = {'device': -1,
        'data': -1,
        'features_num': -1,
        'num_class': -1,
        'hidden': 32,
        'headslayers': 3,
        'dropout_ratio': 0.5,
        'lr': 0.005,
        'weight_decay': 1e-4,
        'beta': 0.4,
        'exInterval': 50,
        'networkType': 0
}

GCN_args = {'device': -1,
        'data': -1,
        'features_num': -1,
        'num_class': -1,
        'hidden': 128,
        'headslayers': 3,
        'dropout_ratio': 0.5,
        'lr': 0.005,
        'weight_decay': 1e-4,
        'beta': 0.4,
        'exInterval': 50,
        'networkType': 1
}

defaultArgsList = [GAT_args, GCN_args]