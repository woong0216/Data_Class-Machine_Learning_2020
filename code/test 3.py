# -*- coding: utf-8 -*-

# DO NOT CHANGE
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
import time
import matplotlib.pyplot as plt

## wkNN
def wkNN(Xtr,ytr,Xts,k,random_state=None):
    # Implement weighted kNN
    # Xtr: training input data set
    # ytr: ouput labels of training set
    # Xts: test data set
    # random_state: random seed
    # return 1-D array containing output labels of test set
    
    
    y_pred = []
    for point in Xts:
        diff = pairwise_distances([point], Xtr)[0]
        index_within_k = np.argsort(diff)[0:k]
        
        if len(np.unique(ytr)) > 2:
            weight = {0:0, 1:0, 2:0}
        else:
            weight = {0:0, 1:0}

        for index in index_within_k:
            if index == index_within_k[0]:
                weight[ytr[index]] = 1
                
            else:
                weight[ytr[index]] = (diff[index_within_k[-1]]-diff[index])/(diff[index_within_k[-1]]-diff[index_within_k[0]]) + weight[ytr[index]] 
        
        y_pred.append(max(weight, key=weight.get))    
        
    
    return y_pred

## PNN
def PNN(Xtr,ytr,Xts,k,random_state=None):
    # Implement PNN
    # Xtr: training input data set
    # ytr: ouput labels of training set
    # Xts: test data set
    # random_state: random seed
    # return 1-D array containing output labels of test set
    
    
    y_pred = []
    point_per_class = {}
    
    if len(np.unique(ytr)) > 2:
        class_num = 3
    else :
        class_num = 2
        
    for class_ in range(0,class_num):
        point_per_class[class_] = Xtr[ytr==class_]
        
    for point in Xts:
        weight = {}
        for class_ in point_per_class:
            weight[class_] = np.sort(pairwise_distances(point_per_class[class_], [point]).reshape(1,-1)[0])[0:k]
            for i in range(1, k+1):
                weight[class_][i-1] = weight[class_][i-1]/i
        
        sum_of_weight_per_class = {}
        for class_ in point_per_class:
            sum_of_weight_per_class[class_] = sum(weight[class_])
            
        y_pred.append(min(sum_of_weight_per_class, key=sum_of_weight_per_class.get))

    
    return y_pred

# 정확도 계산
def accuracy(pred, test):
    total = len(pred)
    number_of_same_element = sum(pred==test)
    
    return round(number_of_same_element/total, 4)



X1,y1=datasets.make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_classes=3, n_clusters_per_class=1, random_state=13)
Xtr1,Xts1, ytr1, yts1=train_test_split(X1,y1,test_size=0.2, random_state=22)

#TODO: Cacluate accuracy with varying k for wkNN and PNN
#TODO: Calculate computation time
#TODO: Draw scatter plot
k=[3,5,7,9,11]
wKNN=[]
pnn=[]
start = time.time()

for a in k:
    y_pred_wknn_1 = wkNN(Xtr1, ytr1, Xts1, a)
    y_pred_pnn_1 = PNN(Xtr1, ytr1, Xts1, a)
    wkNN_value=accuracy(y_pred_wknn_1, yts1)
    pnn_value=accuracy(y_pred_pnn_1, yts1)
    wKNN.append(wkNN_value)
    pnn.append(pnn_value)
time_count= time.time()-start

print("Elapsed time ", time_count)
print('--------------------------------')
print('k            wkNN            PNN')
print(k[0],"          ",wKNN[0],"         ",pnn[0])
print(k[1],"          ",wKNN[1],"         ",pnn[1])
print(k[2],"          ",wKNN[2],"          ",pnn[2])
print(k[3],"          ",wKNN[3],"          ",pnn[3])
print(k[4],"         ",wKNN[4],"         ",pnn[4])


## scatter plot
k=7
wk_TF=(y_pred_wknn_1==yts1)
p_TF=(y_pred_pnn_1==yts1)
plt.rcParams["figure.figsize"] = (10, 10)
val_Xtr1_x=[]
val_Xtr1_y=[]
val_Xts1_x=[]
val_Xts1_y=[]
wk_TF_x=[]
wk_TF_y=[]
p_TF_x=[]
p_TF_y=[]
point_per_train_class = {}
point_per_test_class = {}

for class_ in range(0,3) :
    point_per_train_class[class_] = Xtr1[ytr1==class_]
    point_per_test_class[class_] = Xts1[yts1==class_]
    
for val1, val2 in Xtr1:
    val_Xtr1_x.append(val1)
    val_Xtr1_y.append(val2)
    
for val1, val2 in Xts1:
    val_Xts1_x.append(val1)
    val_Xts1_y.append(val2)
    
for idx in range(len(wk_TF)):
    if wk_TF[idx] == True:
        continue
    else:
        wk_TF_x.append(val_Xtr1_x[idx])
        wk_TF_y.append(val_Xtr1_y[idx])
        
for idx in range(len(p_TF)):
    if p_TF[idx] == True:
        continue
    else:
        p_TF_x.append(val_Xtr1_x[idx])
        p_TF_y.append(val_Xtr1_y[idx])


plt.xlabel('X1')
plt.ylabel('X2')
plt.scatter(val_Xtr1_x,val_Xtr1_y,c='steelblue',marker='o',s=18)
plt.scatter(val_Xts1_x,val_Xts1_y,c='steelblue',marker='x',s=18)
plt.scatter(wk_TF_x,wk_TF_y,facecolors='none',edgecolors='salmon',marker='s',s=50)
plt.scatter(p_TF_x,p_TF_y,facecolors='none',edgecolors='royalblue',marker='d',s=50)
plt.legend(['Train','Test','Misclassifed by wkNN','Miscalssified by PNN'],loc='lower right')
plt.scatter(point_per_train_class[0][: ,0], point_per_train_class[0][:,1], color='purple',marker='o',s=18)
plt.scatter(point_per_train_class[1][: ,0], point_per_train_class[1][:,1], color='green',marker='o',s=18)
plt.scatter(point_per_train_class[2][: ,0], point_per_train_class[2][:,1], color='yellow',marker='o',s=18)
plt.scatter(point_per_test_class[0][: ,0], point_per_test_class[0][:,1], color='purple',marker='x',s=18)
plt.scatter(point_per_test_class[1][: ,0], point_per_test_class[1][:,1], color='green',marker='x',s=18)
plt.scatter(point_per_test_class[2][: ,0], point_per_test_class[2][:,1], color='yellow',marker='x',s=18)



X2,y2=datasets.make_classification(n_samples=1000, n_features=6, n_informative=2, n_redundant=3, n_classes=2, n_clusters_per_class=2, flip_y=0.2,random_state=75)
Xtr2,Xts2, ytr2, yts2=train_test_split(X2,y2,test_size=0.2, random_state=78)

#TODO: Cacluate accuracy with varying k for wkNN and PNN
#TODO: Calculate computation time
#TODO: Draw scatter plot
k=[3,5,7,9,11]
wKNN=[]
pnn=[]
start = time.time()

for a in k:
    y_pred_wknn_2 = wkNN(Xtr2, ytr2, Xts2, a)
    y_pred_pnn_2 = PNN(Xtr2, ytr2, Xts2, a)
    wkNN_value=accuracy(y_pred_wknn_2, yts2)
    pnn_value=accuracy(y_pred_pnn_2, yts2)
    wKNN.append(wkNN_value)
    pnn.append(pnn_value)
time_count= time.time()-start

print("Elapsed time ", time_count)
print('--------------------------------')
print('k            wkNN            PNN')
print(k[0],"          ",wKNN[0],"          ",pnn[0])
print(k[1],"          ",wKNN[1],"         ",pnn[1])
print(k[2],"          ",wKNN[2],"           ",pnn[2])
print(k[3],"          ",wKNN[3],"          ",pnn[3])
print(k[4],"         ",wKNN[4],"          ",pnn[4])

## scatter plot
k=7
wk_TF2=(y_pred_wknn_2==yts2)
p_TF2=(y_pred_pnn_2==yts2)
plt.rcParams["figure.figsize"] = (10, 10)
val_Xtr2_x=[]
val_Xtr2_y=[]
val_Xts2_x=[]
val_Xts2_y=[]
wk_TF_x2=[]
wk_TF_y2=[]
p_TF_x2=[]
p_TF_y2=[]
point_per_train_class = {}
point_per_test_class = {}

for class_ in range(0,2) :
    point_per_train_class[class_] = Xtr2[ytr2==class_]
    point_per_test_class[class_] = Xts2[yts2==class_]
    
for val1, val2, val3, val4, val5, val6 in Xtr2:
    val_Xtr2_x.append(val1)
    val_Xtr2_y.append(val2)
    
for val1, val2, val3, val4, val5, val6 in Xts2:
    val_Xts2_x.append(val1)
    val_Xts2_y.append(val2)
    
for idx in range(len(wk_TF2)):
    if wk_TF2[idx] == True:
        continue
    else:
        wk_TF_x2.append(val_Xtr2_x[idx])
        wk_TF_y2.append(val_Xtr2_y[idx])
        
for idx in range(len(p_TF2)):
    if p_TF2[idx] == True:
        continue
    else:
        p_TF_x2.append(val_Xtr2_x[idx])
        p_TF_y2.append(val_Xtr2_y[idx])


plt.xlabel('X1')
plt.ylabel('X2')
plt.scatter(val_Xtr2_x,val_Xtr2_y,c='steelblue',marker='o',s=18)
plt.scatter(val_Xts2_x,val_Xts2_y,c='steelblue',marker='x',s=18)
plt.scatter(wk_TF_x2,wk_TF_y2,facecolors='none',edgecolors='salmon',marker='s',s=50)
plt.scatter(p_TF_x2,p_TF_y2,facecolors='none',edgecolors='royalblue',marker='d',s=50)
plt.legend(['Train','Test','Misclassifed by wkNN','Miscalssified by PNN'],loc='upper right')
plt.scatter(point_per_train_class[0][: ,0], point_per_train_class[0][:,1], color='purple',marker='o',s=18)
plt.scatter(point_per_train_class[1][: ,0], point_per_train_class[1][:,1], color='yellow',marker='o',s=18)         
plt.scatter(point_per_test_class[0][: ,0], point_per_test_class[0][:,1], color='purple',marker='x',s=18)
plt.scatter(point_per_test_class[1][: ,0], point_per_test_class[1][:,1], color='yellow',marker='x',s=18)
