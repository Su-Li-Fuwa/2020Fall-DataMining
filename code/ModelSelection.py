from network import *
import numpy as np
from hyperopt import fmin, tpe, hp, partial
import copy
import torch

def modelSelection(timer):
    # start dropout estimate
    remindT = timer.record(phase = 1)
    tmpArgs = copy.deepcopy(defaultArgsList[1])
    tmpArgs['dropout_ratio'] = 0.5
    tmpModel = netList[1](tmpArgs).to(tmpArgs['device'])
    _ = tmpModel.myTrain(True, remindT/9)
    del tmpModel
    if _ == -1:     # No enough time for HPO
        return doDropoutSearch(timer)
    return doHPOSearch(timer)

def model_factory(args):
    tmpArgs = copy.deepcopy(defaultArgsList[args['networkType']])
    print(tmpArgs)
    for key in args.keys():
        tmpArgs[key] = args[key]
    if args['networkType'] == 1:
        tmpArgs['hidden'] *= 16
    else:
        tmpArgs['hidden'] = 8*int(args['hidden'])
        tmpArgs['headslayers'] = int(tmpArgs['headslayers'])+1
    print(tmpArgs)
    tmpModel = netList[args['networkType']](tmpArgs).to(tmpArgs["device"])
    valLoss = tmpModel.myTrain(True)
    del tmpModel
    return valLoss

def doHPOSearch(timer):
    timeLimit = timer.record(phase = 1.1)
    print(int(timeLimit))
    #input()
    GCNSearchSpace = {
        'networkType': hp.randint('networkType', len(netList)),
        'hidden': hp.randint('hidden', 1, 9),    # (16*8=128, 16*hidden)
        'headslayers': hp.randint('headslayers', 1, 4),      # 
        'dropout_ratio': hp.uniform('dropout_ratio', 0.2, 0.8),
        'lr': hp.uniform('lr', 1e-4, 1e-2),
        }
    
    algo = partial(tpe.suggest, n_startup_jobs=1)
    best = fmin(model_factory, GCNSearchSpace, algo=algo, timeout = int(timeLimit), pass_expr_memo_ctrl=None)
    bestArgs = copy.deepcopy(defaultArgsList[best['networkType']])
    for key in best.keys():
        bestArgs[key] = best[key]
    if best['networkType'] == 1:
        bestArgs['hidden'] *= 16
    else:
        bestArgs['hidden'] = 8*int(best['hidden'])
        bestArgs['headslayers'] = int(best['headslayers'])+1
    return netList[best['networkType']], bestArgs

def doDropoutSearch(timer):
    timer.record(phase = 1.2)
    dropList = [0]
    notTimeOut = True
    minLoss = 100
    i = 0

    tmpModel = [netList[j](defaultArgsList[j]).to(defaultArgsList[j]['device']) for j in range(len(netList))]
    while i < len(dropList):
        for j in range(len(netList)):
            tmpArgs = copy.deepcopy(defaultArgsList[j])
            tmpArgs['dropout_ratio'] = 0.2 + dropList[i]
            tmpModel[j].resetMetaData(tmpArgs)
            _ = tmpModel[j].myTrain(True, timer.record())
            if _ < 0:  break  
            if _ < minLoss:
                selectedModel = netList[j]
                selectedPara =  tmpArgs
                minLoss = _
            print("Dropout: %f\t Net: %s\t loss: %s\t"%(0.2 + dropList[i], netList[j], _))
            
        if _ < 0:  break
        if i == 0:  # first estimate 
            dropList = timer.getDropList(len(netList))
        i += 1
    return selectedModel, selectedPara