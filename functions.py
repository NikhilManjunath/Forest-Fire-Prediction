#Importing the required libraries
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import *
from sklearn.feature_selection import SequentialFeatureSelector

#Supress Sklearn Warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

####################### Functions ############################
#Utility Function to Compute Metrics to Calculate False Positives
def compute_metrics(preds,labels):
    metrics={}
    TP=0
    TN=0
    FP=0
    FN=0
    for i in range(preds.size):
        if preds[i]==0 :
            if labels[i]==0:
                TN+=1
            if labels[i]==1:
                FN+=1
        else:
            if labels[i]==0:
                FP+=1
            if labels[i]==1:
                TP+=1
    metrics['false_negatives'] = FN
    metrics['accuracy'] = np.round((TP+TN)/(TP+TN+FP+FN),2)
    precision = np.round(TP/(TP+FP),2)
    metrics['precision'] = precision
    recall = np.round(TP/(TP+FN),2)
    metrics['recall'] =  recall
    metrics['f1_score'] = 2*np.round(precision*recall/(precision+recall),2)
    return metrics

#Correlation Heatmap
def correlation_heatmap(X):
    """ Function to plot Correlation coefficient heatmap based on input data """
    corr = X.corr()
    sns.set(rc={'figure.figsize':(11.7,8.27)})
    sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns, annot=True)

#PCA Analysis Function
def PCA_analysis(train_features):
    pca_full = PCA()
    pca_full.fit(train_features)
    cum_sum = np.cumsum(pca_full.explained_variance_ratio_)
    plt.figure()
    plt.title('PCA Skree Plot')
    plt.xlabel("No. of Dimensions")
    plt.ylabel("Eigen values")
    plt.plot(pca_full.explained_variance_ratio_)
    plt.xlim([-1,20])
    plt.show()

    plt.figure()
    plt.title('PCA Cummulated Variance v/s No. of. Dimensions')
    plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
    plt.xlim([-1,20])
    plt.xlabel("No. of Dimensions")
    plt.ylabel("PCA Variance (Cumulated)")
    plt.show()
    
#PCA Function
def PCA_func(train,test,n_components):
    pca = PCA(n_components=n_components)
    train_PCA = pca.fit_transform(train)
    train_PCA = pd.DataFrame(train_PCA)
    test_PCA = pca.transform(test)
    test_PCA = pd.DataFrame(test_PCA)
    return train_PCA,test_PCA

#SFS Function
def seq_feat_selector(train,labels,n_max):
    folds,classes = k_fold(train,labels)
    train_features_folds,train_classes_folds,_,_ = cross_val(folds,classes)
    knn = KNeighborsClassifier()
    sv_c = SVC(kernel='linear')
    ridge = RidgeClassifier()
    logistic = LogisticRegression(penalty='l2')
    feat_model= []
    for model in [knn,sv_c,ridge,logistic]:  
        feat_per_fold = []
        for i in range(len(train_classes_folds)):
            sfs = SequentialFeatureSelector(model,n_features_to_select=18)
            sfs.fit(train_features_folds[i],train_classes_folds[i])
            feat_per_fold.append(sfs.get_support(indices=True))
        feat_model.append(feat_per_fold)
    unique,counts = np.unique(feat_model,return_counts=True)
    res = unique[np.argsort(counts)]
    cols = res[-n_max:]
    return train.columns[cols]

#K-fold Cross Validation Functions
def k_fold(features,labels):
    folds = {}
    class_folds = {}
    for i in range(0,6):
        folds[i] = features[30*i:30*(i+1)]
        class_folds[i] = labels[30*i:30*(i+1)]
    return folds,class_folds
    
def cross_val(folds,classes):
    train_features_folds = []
    val_features_folds = []
    train_classes_folds = []
    val_classes_folds = []
    keys = set(folds.keys())
    for i in range(len(folds)):
        val = folds[i].copy(deep=True)
        val_classes = classes[i].copy(deep=True)
        val.drop(val.tail(6).index,inplace=True)
        val_classes.drop(val_classes.tail(6).index,inplace=True)
        
        train = []
        train_class = []
        
        if i != 0:
            excludes = set([i, i-1])
        
            val_prev = folds[i-1].copy(deep=True)
            val_prev.drop(val_prev.tail(6).index,inplace=True)
            val_prev_class = classes[i-1].copy(deep=True)
            val_prev_class.drop(val_prev_class.tail(6).index,inplace=True)
            
            train.append(val_prev)
            train_class.append(val_prev_class)
            
        else:
            excludes = set([i])
            
        for key in keys.difference(excludes):
            train.append(folds[key])
            train_class.append(classes[key])
            
            
        train = np.vstack(train)
        train_class = np.hstack(train_class)
        train_features_folds.append(train)
        train_classes_folds.append(train_class)
        val_features_folds.append(val)
        val_classes_folds.append(val_classes)
    return train_features_folds,train_classes_folds,val_features_folds,val_classes_folds 


#Trivial Model
def trivial(train,test):
    N=train.shape[0]
    N0 = train[train['Classes']==0].shape[0]
    N1 = train[train['Classes']==1].shape[0]
    p0 = N0/N
    p1 = N1/N
    preds = []
    for i in range(test.shape[0]):
        preds.append(random.choices([0,1],[p0,p1]))
    preds = np.asarray(preds)
    labels =  test['Classes'].to_numpy()
    metrics = compute_metrics(preds,labels)
    print(classification_report(labels,preds))
    print(metrics)
    sns.heatmap(confusion_matrix(labels,preds),annot=True)

#Baseline Nearest Means Model
def nearest_means(train, test):
    #Grouping the Data Classwise
    train = train.to_numpy()
    test = test.to_numpy()
    group_1 = train[np.where(train[:,-1] == 0)]
    group_1 = group_1[:,1:-1]
    group_2 = train[np.where(train[:,-1] == 1)]
    group_2 = group_2[:,1:-1]

    #Finding the mean for each group
    group_1_mean = group_1.mean(axis=0)
    group_2_mean = group_2.mean(axis=0)

    #Seperating the inputs and the labels
    train_data = train[:,1:-1]
    train_labels = train[:,-1]
    test_data = test[:,1:-1]
    test_labels = test[:,-1]
    
    #Array to store predictions
    train_predictions = []
    test_predictions = []
    
    #Running the nearest means model
    for i in train_data:
      train_dist1 = np.linalg.norm(i - group_1_mean)
      train_dist2 = np.linalg.norm(i - group_2_mean)
      if train_dist1 < train_dist2:
        train_predictions.append(0)
      elif train_dist2 < train_dist1:
        train_predictions.append(1)
        
    for i in test_data:
      test_dist1 = np.linalg.norm(i - group_1_mean)
      test_dist2 = np.linalg.norm(i - group_2_mean)
      if test_dist1 < test_dist2:
        test_predictions.append(0)
      elif test_dist2 < test_dist1:
        test_predictions.append(1)

    #Calculating error rate
    train_error_count = 0
    for i in range(0,train.shape[0]):
      if train_predictions[i] != train_labels[i]:
        train_error_count += 1
        
    train_accuracy = (1-(train_error_count/train.shape[0]))*100
    # print('Accuracy: Nearest Means - Train Dataset',train_accuracy, '%')
        
    test_error_count = 0
    for i in range(0,test.shape[0]):
      if test_predictions[i] != test_labels[i]:
        test_error_count += 1

    test_accuracy = (1-test_error_count/test.shape[0])*100
    print('Accuracy of Nearest Means on the Test Dataset is', np.round(test_accuracy,2), '%')
    test_predictions = np.array(test_predictions,dtype=int)
    test_labels = np.array(test_labels,dtype=int)
    print(classification_report(test_labels,test_predictions))

    print(compute_metrics(test_predictions,test_labels))
    sns.heatmap(confusion_matrix(test_labels,test_predictions),annot=True)