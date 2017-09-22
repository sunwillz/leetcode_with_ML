# -*- coding: utf-8 -*-

## 评价标准

import numpy as np


def accuracy(y_true, y_pred):
    if len(y_true)!=len(y_pred):
        print "the data are incorrect!!!"
        return
    n = len(y_true)
    count = 0
    for i in range(n):
        if(y_pred[i]==y_true[i]):count += 1

    return float(count)/n


def rmse(y_true,y_pred):
    if len(y_true)!=len(y_pred):
        print "the data are incorrect!!!"
        return
    score = 0.0
    for i in range(len(y_true)):
        score += (y_pred[i]-y_true[i])*(y_pred[i]-y_true[i])

    return np.sqrt(score/len(y_true))