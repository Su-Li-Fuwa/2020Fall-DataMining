import numpy as np 
import time

def softmax(x, axis=1):
    row_max = x.max(axis=axis)
    row_max=row_max.reshape(-1, 1)
    x = x - row_max
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s

class Timer:
    def __init__(self, time_budget,  percent1 = 0.2, percent2 = 0.6):
        self.totalTime = time_budget
        self.lastRecord = time.time()
        self.timeRecord = []
        self.phase = -1
        self.startT = -1
        self.end1, self.end2 = time_budget*percent1, time_budget*(percent1+percent2)
        self.trained = -1
        
    
    def record(self, phase = -1):
        self.timeRecord.append(time.time())
        if (phase == 0):
            self.startT = self.timeRecord[-1]
            self.phase = phase
            return True
        elif (phase == 0.1):
            remindT = self.end1 - (self.timeRecord[-1] - self.startT)
            takes1000 = self.timeRecord[-1] - self.timeRecord[-2]
            self.phase = phase
            return int((remindT/takes1000)*1000)
        elif (phase == 1 or phase == 1.1):
            remindT = self.end2 - (self.timeRecord[-1] - self.startT)
            return remindT
        elif (phase == 2 or phase == 1.2):
            self.trained = 0
            self.phase = phase
            return True
        else:
            if (self.phase == 1.2):
                remindT = self.end2 - (self.timeRecord[-1] - self.startT)
                print("phase 1\t remind: ", remindT)
                return remindT
            elif (self.phase == 2):
                remindT = self.totalTime - 5 - (self.timeRecord[-1] - self.startT)
                return remindT
            elif (self.phase == 0.1):
                remindT = self.end1 - (self.timeRecord[-1] - self.startT)
                if remindT <= 0:    return True
                return False
            else:
                return True
            
    
    def getDropList(self, nEstmate):
        interval = self.timeRecord[-1] - self.timeRecord[-1-nEstmate]
        remindT = self.end2 - (self.timeRecord[-1] - self.startT)
        length, dropList = int(remindT/interval), [0]
        repeat = (int((length-1)/3) >= 2)
        length = min(length, 3)
        if repeat:
            dropList.append(0)
        #print(remindT, interval, length)
        #print("************************")
        for i in range(1,length+1):
            dropList.append(0.6*i/length)
            if repeat:
                dropList.append(0.6*i/length)
        return dropList