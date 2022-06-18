import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import TruncatedSVD
import random
# import seaborn as sns
import os.path as path
import os
# import matplotlib
# import matplotlib.font_manager
# import matplotlib.pyplot as plt # graphs plotting
# import Bio
from Bio import SeqIO # some BioPython that will come in handy
#matplotlib inline

from itertools import cycle

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score


# from matplotlib import rc
# # for Arial typefont
# matplotlib.rcParams['font.family'] = 'Arial'

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn import svm

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler  
from sklearn.neural_network import MLPClassifier 
from sklearn.metrics import classification_report, confusion_matrix 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from pandas import DataFrame

from sklearn.model_selection import KFold 
from sklearn.model_selection import RepeatedKFold

from sklearn.metrics import confusion_matrix

from numpy import mean


import itertools
from itertools import product

import csv

from sklearn.model_selection import ShuffleSplit # or StratifiedShuffleSplit

from sklearn.decomposition import KernelPCA

import timeit

import random
import matplotlib.pyplot as plt

from tsai.all import *
computer_setup()

import sys
sys.stdout = open('./output.txt', 'w')

print("Packages Loading done!!")

seq_data = np.load("/olga-data0/Sarwan/Sequence_to_time_series_to_image_classification/Dataset/seq_data_7000.npy")
attribute_data = np.load("/olga-data0/Sarwan/Sequence_to_time_series_to_image_classification/Dataset/seq_data_variant_names_7000.npy")


# 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
#        'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y'

total_int_seq = []

for i in range(len(seq_data)):
    dnaSeq = list(seq_data[i])
    res = [item.replace('A', '11') for item in dnaSeq]
    res = [item.replace('C', '10') for item in res]
    res = [item.replace('D', '9') for item in res]
    res = [item.replace('E', '8') for item in res]    
    res = [item.replace('F', '7') for item in res]
    res = [item.replace('G', '6') for item in res]
    res = [item.replace('H', '5') for item in res]    
    res = [item.replace('I', '4') for item in res]
    res = [item.replace('K', '3') for item in res]
    res = [item.replace('L', '2') for item in res]    
    res = [item.replace('M', '1') for item in res]
    res = [item.replace('N', '-1') for item in res]
    res = [item.replace('P', '-2') for item in res]    
    res = [item.replace('Q', '-3') for item in res]
    res = [item.replace('R', '-4') for item in res]
    res = [item.replace('S', '-5') for item in res]    
    res = [item.replace('T', '-6') for item in res]
    res = [item.replace('V', '-7') for item in res]
    res = [item.replace('W', '-8') for item in res]    
    res = [item.replace('X', '-9') for item in res]
    res = [item.replace('Y', '-10') for item in res]
    
    data = []
    for i in range(len(res)):
        data.append(float(res[i]))
    # print(data)
    
    total_int_seq.append(data)


# attribute_data = np.load("E:/RA/IJCAI/Dataset/Original/second_seq_data_variant_names_7000.npy")
attr_new = []
for i in range(len(attribute_data)):
    aa = str(attribute_data[i]).replace("[","")
    aa_1 = aa.replace("]","")
    aa_2 = aa_1.replace("\'","")
    attr_new.append(aa_2)
    
# y_val = []

# for i in range(len(total_int_seq)):
#     if i%2==0:
#         y_val.append(1)
#     else:
#         y_val.append(2)

unique_hst = list(np.unique(attr_new))

int_hosts = []
for ind_unique in range(len(attr_new)):
    variant_tmp = attr_new[ind_unique]
    ind_tmp = unique_hst.index(variant_tmp)
    int_hosts.append(ind_tmp)
    
print("Attribute data preprocessing Done")

def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

    unique_class = set(actual_class)
    roc_auc_dict = {}
    for per_class in unique_class:
        #creating a list of all the classes except the current class 
        other_class = [x for x in unique_class if x != per_class]

        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        #using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc
    
    
    check = pd.DataFrame(roc_auc_dict.items())
    return mean(check)


def fun_accrs(actual_class, pred_class):
    y_test = actual_class[:]
    y_pred = pred_class[:]

    dt_acc = metrics.accuracy_score(y_test, y_pred)
    dt_prec = metrics.precision_score(y_test, y_pred,average='weighted')
    dt_recall = metrics.recall_score(y_test, y_pred,average='weighted')
    dt_f1_weighted = metrics.f1_score(y_test, y_pred,average='weighted')
    dt_f1_macro = metrics.f1_score(y_test, y_pred,average='macro')
    dt_f1_micro = metrics.f1_score(y_test, y_pred,average='micro')
    macro_roc_auc_ovo = roc_auc_score_multiclass(y_test, y_pred, average='macro')

    check = [dt_acc,dt_prec,dt_recall,dt_f1_weighted,dt_f1_macro,dt_f1_micro,macro_roc_auc_ovo[1]]

    return(check)
    

com_sum_full_data = []

for i in range(len(total_int_seq)):
    com_sum_full_data.append(np.cumsum(total_int_seq[i])) #cumulative sum

y_val = int_hosts[:]


Xx = np.array(com_sum_full_data)
y = np.array(y_val)

aa = Xx.reshape((Xx.shape[0],1, Xx.shape[1]))

X = aa[:]

splits_new_2 = get_splits(np.array(y), valid_size=.10, random_state=23, stratify=True)

X_train = X[splits_new_2[0]]
y_train = y[splits_new_2[0]]
X_test = X[splits_new_2[1]]
y_test = y[splits_new_2[1]]

X = X_train[:]
y = y_train[:]



print("Train-Test Split Logic Done")

unique_labels = len(np.unique(y_val))
tfms = [None, Categorize()]
bts = [[TSNormalize(), TSToPlot()], 
       [TSNormalize(), TSToMat(cmap='viridis')],
       [TSNormalize(), TSToGADF(cmap='spring')],
       [TSNormalize(), TSToGASF(cmap='summer')],
       [TSNormalize(), TSToMTF(cmap='autumn')],
       [TSNormalize(), TSToRP(cmap='winter')]]
btns = ['Plot', 'Mat', 'GADF', 'GASF', 'MTF', 'RP']

splits_new = get_splits(np.array(y), valid_size=.11, random_state=23, stratify=True)

for i, (bt, btn) in enumerate(zip(bts, btns)):
#     yy = np.random.randint(0, 2, 60)
    
    dsets = TSDatasets(X, y, tfms=tfms, splits=splits_new)
    dls = TSDataLoaders.from_dsets(dsets.train,
                                   dsets.valid,
                                   bs=[64, 128],
                                   batch_tfms=bt,
                                   shuffle=False)
    xb, yb = dls.train.one_batch()
    print(f'\n\ntfm: TSTo{btn} - batch shape: {xb.shape}')
    xb[0].show()
    plt.show()
    
    
tfms = [None, Categorize()]
batch_tfms = [TSStandardize()]
# dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
dsets = TSDatasets(X, y, tfms=tfms, splits=splits_new)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=batch_tfms)
dls.show_batch()


epochs = 100

print("Number of Epochs = ",epochs)

model = create_model(InceptionTime, dls=dls)
# learn = Learner(dls, model, metrics=accuracy, cbs=ShowGraphCallback())
learn = Learner(dls, model, metrics=accuracy)
start = time.time()
learn.fit_one_cycle(epochs, lr_max=1e-3)
print(f"\ntraining time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
learn.plot_metrics()

vals = learn.recorder.values[-1]
results= [vals[0], vals[1], vals[2]]
final_results = pd.DataFrame(results).T
final_results.columns = ["train loss","valid loss","accuracy"]
display(final_results)

########################## Evaluation Metrics (Start) ###########################    
test_probas3, test_targets3, test_preds3, test_losses3 = learn.get_X_preds(X_test, y_test, with_loss=True, with_decoded=True)

# test_targets3
aa = pd.DataFrame(list(test_preds3))
# aa.drop("]", axis=0)
aa.columns = ["name"]
aa = aa.loc[aa["name"] != "["]
aa = aa.loc[aa["name"] != ","]
aa = aa.loc[aa["name"] != ","]
aa = aa.loc[aa["name"] != " "]
aa = aa.loc[aa["name"] != "]"]
# aa
# test_preds3

asd = (aa.values.tolist())
final_pred_vals = []
for i in range(len(asd)):
    temp = asd[i]
    temp_1 = str(temp).replace("\'","")
    temp_2 = str(temp_1).replace("[","")
    temp_3 = int(str(temp_2).replace("]",""))
    final_pred_vals.append(temp_3)
    
tmp_lst = list(test_targets3)
final_org_vals = []
for i in range(len(tmp_lst)):
    temp = str(tmp_lst[i])
    temp_1 = str(temp).replace("TensorCategory(","")
    temp_2 = int(str(temp_1).replace(")",""))
    final_org_vals.append(temp_2)


dt_table = []
dt_return = fun_accrs(final_org_vals,final_pred_vals)
dt_table.append(dt_return)

dt_table_final = DataFrame(dt_table, columns=["Accuracy","Precision","Recall",
                                                    "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])

print(dt_table_final)
########################## Evaluation Metrics (End) ###########################

################################ TSToPlot (Start) ###################################
print("TSToPlot Start")

tfms = [None, Categorize()]
batch_tfms = [TSNormalize(), TSToPlot()]
# dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
dsets = TSDatasets(X, y, tfms=tfms, splits=splits_new)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=batch_tfms)
dls.show_batch()

model = create_model(xresnet34, dls=dls)
learn = Learner(dls, model, metrics=accuracy)
start = time.time()
learn.fit_one_cycle(epochs, lr_max=1e-3)
print(f"\ntraining time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
learn.plot_metrics()

vals = learn.recorder.values[-1]
results= [vals[0], vals[1], vals[2]]
# results.sort_values(by='accuracy', ascending=False, ignore_index=True, inplace=True)
# clear_output()
# final_results = np.array(results).to_frame().T
final_results = pd.DataFrame(results).T
final_results.columns = ["train loss","valid loss","accuracy"]
display(final_results)

########################## Evaluation Metrics (Start) ###########################    
test_probas3, test_targets3, test_preds3, test_losses3 = learn.get_X_preds(X_test, y_test, with_loss=True, with_decoded=True)

# test_targets3
aa = pd.DataFrame(list(test_preds3))
# aa.drop("]", axis=0)
aa.columns = ["name"]
aa = aa.loc[aa["name"] != "["]
aa = aa.loc[aa["name"] != ","]
aa = aa.loc[aa["name"] != ","]
aa = aa.loc[aa["name"] != " "]
aa = aa.loc[aa["name"] != "]"]
# aa
# test_preds3

asd = (aa.values.tolist())
final_pred_vals = []
for i in range(len(asd)):
    temp = asd[i]
    temp_1 = str(temp).replace("\'","")
    temp_2 = str(temp_1).replace("[","")
    temp_3 = int(str(temp_2).replace("]",""))
    final_pred_vals.append(temp_3)
    
tmp_lst = list(test_targets3)
final_org_vals = []
for i in range(len(tmp_lst)):
    temp = str(tmp_lst[i])
    temp_1 = str(temp).replace("TensorCategory(","")
    temp_2 = int(str(temp_1).replace(")",""))
    final_org_vals.append(temp_2)


dt_table = []
dt_return = fun_accrs(final_org_vals,final_pred_vals)
dt_table.append(dt_return)

dt_table_final = DataFrame(dt_table, columns=["Accuracy","Precision","Recall",
                                                    "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])

print(dt_table_final)
########################## Evaluation Metrics (End) ###########################

################################ TSToPlot (Ends) ###################################


################################ Time Series to Mat (Start) ###################################
print("Time Series to Mat Start")

# https://timeseriesai.github.io/tsai/data.validation.html
tfms = [None, Categorize()]
# batch_tfms = [TSNormalize(), TSToMat()]
batch_tfms = [TSNormalize(), TSToMat(cmap='viridis')]
# dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
# dsets = TSDatasets(X, y, tfms=tfms, splits=None)
# yy = np.random.randint(0, 2, 60)
dsets = TSDatasets(X, y, tfms=tfms,splits=splits_new)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=batch_tfms)
dls.show_batch()

tfms = [None, Categorize()]
batch_tfms = [TSNormalize(), TSToMat()]
# batch_tfms = [TSNormalize(), TSToMat(cmap='viridis')]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits_new)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=batch_tfms)
# dls.show_batch()
xb, yb = dls.train.one_batch()
# print(f'\n\ntfm: TSTo{btn} - batch shape: {xb.shape}')
xb[0].show()
len(xb)

model = create_model(xresnet34, dls=dls, pretrained=True)
print("Pretrained XResnet34 model")
learn = Learner(dls, model, metrics=accuracy)
start = time.time()
learn.fit_one_cycle(epochs, lr_max=1e-3)
print(f"\ntraining time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
learn.plot_metrics()

vals = learn.recorder.values[-1]
results= [vals[0], vals[1], vals[2]]
# results.sort_values(by='accuracy', ascending=False, ignore_index=True, inplace=True)
# clear_output()
# final_results = np.array(results).to_frame().T
final_results = pd.DataFrame(results).T
final_results.columns = ["train loss","valid loss","accuracy"]
display(final_results)

########################## Evaluation Metrics (Start) ###########################    
test_probas3, test_targets3, test_preds3, test_losses3 = learn.get_X_preds(X_test, y_test, with_loss=True, with_decoded=True)

# test_targets3
aa = pd.DataFrame(list(test_preds3))
# aa.drop("]", axis=0)
aa.columns = ["name"]
aa = aa.loc[aa["name"] != "["]
aa = aa.loc[aa["name"] != ","]
aa = aa.loc[aa["name"] != ","]
aa = aa.loc[aa["name"] != " "]
aa = aa.loc[aa["name"] != "]"]
# aa
# test_preds3

asd = (aa.values.tolist())
final_pred_vals = []
for i in range(len(asd)):
    temp = asd[i]
    temp_1 = str(temp).replace("\'","")
    temp_2 = str(temp_1).replace("[","")
    temp_3 = int(str(temp_2).replace("]",""))
    final_pred_vals.append(temp_3)
    
tmp_lst = list(test_targets3)
final_org_vals = []
for i in range(len(tmp_lst)):
    temp = str(tmp_lst[i])
    temp_1 = str(temp).replace("TensorCategory(","")
    temp_2 = int(str(temp_1).replace(")",""))
    final_org_vals.append(temp_2)


dt_table = []
dt_return = fun_accrs(final_org_vals,final_pred_vals)
dt_table.append(dt_return)

dt_table_final = DataFrame(dt_table, columns=["Accuracy","Precision","Recall",
                                                    "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])

print(dt_table_final)
########################## Evaluation Metrics (End) ###########################

model = create_model(xresnet34, dls=dls) # by default xresnet models are pretrained=False
print("newly trained xresnet34 model")
learn = Learner(dls, model, metrics=accuracy)
start = time.time()
learn.fit_one_cycle(epochs, lr_max=1e-3)
print(f"\ntraining time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
learn.plot_metrics()

vals = learn.recorder.values[-1]
results= [vals[0], vals[1], vals[2]]
final_results = pd.DataFrame(results).T
final_results.columns = ["train loss","valid loss","accuracy"]
display(final_results)

########################## Evaluation Metrics (Start) ###########################    
test_probas3, test_targets3, test_preds3, test_losses3 = learn.get_X_preds(X_test, y_test, with_loss=True, with_decoded=True)

# test_targets3
aa = pd.DataFrame(list(test_preds3))
# aa.drop("]", axis=0)
aa.columns = ["name"]
aa = aa.loc[aa["name"] != "["]
aa = aa.loc[aa["name"] != ","]
aa = aa.loc[aa["name"] != ","]
aa = aa.loc[aa["name"] != " "]
aa = aa.loc[aa["name"] != "]"]
# aa
# test_preds3

asd = (aa.values.tolist())
final_pred_vals = []
for i in range(len(asd)):
    temp = asd[i]
    temp_1 = str(temp).replace("\'","")
    temp_2 = str(temp_1).replace("[","")
    temp_3 = int(str(temp_2).replace("]",""))
    final_pred_vals.append(temp_3)
    
tmp_lst = list(test_targets3)
final_org_vals = []
for i in range(len(tmp_lst)):
    temp = str(tmp_lst[i])
    temp_1 = str(temp).replace("TensorCategory(","")
    temp_2 = int(str(temp_1).replace(")",""))
    final_org_vals.append(temp_2)


dt_table = []
dt_return = fun_accrs(final_org_vals,final_pred_vals)
dt_table.append(dt_return)

dt_table_final = DataFrame(dt_table, columns=["Accuracy","Precision","Recall",
                                                    "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])

print(dt_table_final)
########################## Evaluation Metrics (End) ###########################

################################ Time Series to Mat (Ends) ###################################

################################ Time Series to GADF (Start) ###################################
print("Time Series to GADF Start")

tfms = [None, Categorize()]
batch_tfms = [TSNormalize(), TSToGADF()]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits_new)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=batch_tfms)
dls.show_batch()

model = create_model(xresnet34, dls=dls)
learn = Learner(dls, model, metrics=accuracy)
start = time.time()
learn.fit_one_cycle(epochs, lr_max=1e-3)
print(f"\ntraining time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
learn.plot_metrics()

vals = learn.recorder.values[-1]
results= [vals[0], vals[1], vals[2]]
final_results = pd.DataFrame(results).T
final_results.columns = ["train loss","valid loss","accuracy"]
display(final_results)

########################## Evaluation Metrics (Start) ###########################    
test_probas3, test_targets3, test_preds3, test_losses3 = learn.get_X_preds(X_test, y_test, with_loss=True, with_decoded=True)

# test_targets3
aa = pd.DataFrame(list(test_preds3))
# aa.drop("]", axis=0)
aa.columns = ["name"]
aa = aa.loc[aa["name"] != "["]
aa = aa.loc[aa["name"] != ","]
aa = aa.loc[aa["name"] != ","]
aa = aa.loc[aa["name"] != " "]
aa = aa.loc[aa["name"] != "]"]
# aa
# test_preds3

asd = (aa.values.tolist())
final_pred_vals = []
for i in range(len(asd)):
    temp = asd[i]
    temp_1 = str(temp).replace("\'","")
    temp_2 = str(temp_1).replace("[","")
    temp_3 = int(str(temp_2).replace("]",""))
    final_pred_vals.append(temp_3)
    
tmp_lst = list(test_targets3)
final_org_vals = []
for i in range(len(tmp_lst)):
    temp = str(tmp_lst[i])
    temp_1 = str(temp).replace("TensorCategory(","")
    temp_2 = int(str(temp_1).replace(")",""))
    final_org_vals.append(temp_2)


dt_table = []
dt_return = fun_accrs(final_org_vals,final_pred_vals)
dt_table.append(dt_return)

dt_table_final = DataFrame(dt_table, columns=["Accuracy","Precision","Recall",
                                                    "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])

print(dt_table_final)
########################## Evaluation Metrics (End) ###########################

################################ Time Series to GADF (Ends) ###################################

################################ Time Series to GASF (Start) ###################################
print("Time Series to GASF Start")

tfms = [None, Categorize()]
batch_tfms = [TSNormalize(), TSToGASF()]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits_new)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=batch_tfms)
dls.show_batch()

model = create_model(xresnet34, dls=dls)
learn = Learner(dls, model, metrics=accuracy)
start = time.time()
learn.fit_one_cycle(epochs, lr_max=1e-3)
print(f"\ntraining time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
learn.plot_metrics()

vals = learn.recorder.values[-1]
results= [vals[0], vals[1], vals[2]]
final_results = pd.DataFrame(results).T
final_results.columns = ["train loss","valid loss","accuracy"]
display(final_results)

########################## Evaluation Metrics (Start) ###########################    
test_probas3, test_targets3, test_preds3, test_losses3 = learn.get_X_preds(X_test, y_test, with_loss=True, with_decoded=True)

# test_targets3
aa = pd.DataFrame(list(test_preds3))
# aa.drop("]", axis=0)
aa.columns = ["name"]
aa = aa.loc[aa["name"] != "["]
aa = aa.loc[aa["name"] != ","]
aa = aa.loc[aa["name"] != ","]
aa = aa.loc[aa["name"] != " "]
aa = aa.loc[aa["name"] != "]"]
# aa
# test_preds3

asd = (aa.values.tolist())
final_pred_vals = []
for i in range(len(asd)):
    temp = asd[i]
    temp_1 = str(temp).replace("\'","")
    temp_2 = str(temp_1).replace("[","")
    temp_3 = int(str(temp_2).replace("]",""))
    final_pred_vals.append(temp_3)
    
tmp_lst = list(test_targets3)
final_org_vals = []
for i in range(len(tmp_lst)):
    temp = str(tmp_lst[i])
    temp_1 = str(temp).replace("TensorCategory(","")
    temp_2 = int(str(temp_1).replace(")",""))
    final_org_vals.append(temp_2)


dt_table = []
dt_return = fun_accrs(final_org_vals,final_pred_vals)
dt_table.append(dt_return)

dt_table_final = DataFrame(dt_table, columns=["Accuracy","Precision","Recall",
                                                    "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])

print(dt_table_final)
########################## Evaluation Metrics (End) ###########################

################################ Time Series to GASF (Ends) ###################################

################################ Time Series to MTF (Start) ###################################
print("Time Series to MTF Start")

tfms = [None, Categorize()]
batch_tfms = [TSNormalize(), TSToMTF()]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits_new)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=batch_tfms)
dls.show_batch()

model = create_model(xresnet34, dls=dls)
learn = Learner(dls, model, metrics=accuracy)
start = time.time()
learn.fit_one_cycle(epochs, lr_max=1e-3)
print(f"\ntraining time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
learn.plot_metrics()

vals = learn.recorder.values[-1]
results= [vals[0], vals[1], vals[2]]
final_results = pd.DataFrame(results).T
final_results.columns = ["train loss","valid loss","accuracy"]
display(final_results)

########################## Evaluation Metrics (Start) ###########################    
test_probas3, test_targets3, test_preds3, test_losses3 = learn.get_X_preds(X_test, y_test, with_loss=True, with_decoded=True)

# test_targets3
aa = pd.DataFrame(list(test_preds3))
# aa.drop("]", axis=0)
aa.columns = ["name"]
aa = aa.loc[aa["name"] != "["]
aa = aa.loc[aa["name"] != ","]
aa = aa.loc[aa["name"] != ","]
aa = aa.loc[aa["name"] != " "]
aa = aa.loc[aa["name"] != "]"]
# aa
# test_preds3

asd = (aa.values.tolist())
final_pred_vals = []
for i in range(len(asd)):
    temp = asd[i]
    temp_1 = str(temp).replace("\'","")
    temp_2 = str(temp_1).replace("[","")
    temp_3 = int(str(temp_2).replace("]",""))
    final_pred_vals.append(temp_3)
    
tmp_lst = list(test_targets3)
final_org_vals = []
for i in range(len(tmp_lst)):
    temp = str(tmp_lst[i])
    temp_1 = str(temp).replace("TensorCategory(","")
    temp_2 = int(str(temp_1).replace(")",""))
    final_org_vals.append(temp_2)


dt_table = []
dt_return = fun_accrs(final_org_vals,final_pred_vals)
dt_table.append(dt_return)

dt_table_final = DataFrame(dt_table, columns=["Accuracy","Precision","Recall",
                                                    "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])

print(dt_table_final)
########################## Evaluation Metrics (End) ###########################

################################ Time Series to MTF (Ends) ###################################

################################ Time Series to RP (Start) ###################################
print("Time Series to RP Start")

tfms = [None, Categorize()]
batch_tfms = [TSNormalize(by_sample=True, range=(0,1)), TSToRP()]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits_new)
dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=batch_tfms)
dls.show_batch()

model = create_model(xresnet34, dls=dls)
learn = Learner(dls, model, metrics=accuracy)
start = time.time()
learn.fit_one_cycle(epochs, lr_max=1e-3)
print(f"\ntraining time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")
learn.plot_metrics()

vals = learn.recorder.values[-1]
results= [vals[0], vals[1], vals[2]]
final_results = pd.DataFrame(results).T
final_results.columns = ["train loss","valid loss","accuracy"]
display(final_results)

########################## Evaluation Metrics (Start) ###########################    
test_probas3, test_targets3, test_preds3, test_losses3 = learn.get_X_preds(X_test, y_test, with_loss=True, with_decoded=True)

# test_targets3
aa = pd.DataFrame(list(test_preds3))
# aa.drop("]", axis=0)
aa.columns = ["name"]
aa = aa.loc[aa["name"] != "["]
aa = aa.loc[aa["name"] != ","]
aa = aa.loc[aa["name"] != ","]
aa = aa.loc[aa["name"] != " "]
aa = aa.loc[aa["name"] != "]"]
# aa
# test_preds3

asd = (aa.values.tolist())
final_pred_vals = []
for i in range(len(asd)):
    temp = asd[i]
    temp_1 = str(temp).replace("\'","")
    temp_2 = str(temp_1).replace("[","")
    temp_3 = int(str(temp_2).replace("]",""))
    final_pred_vals.append(temp_3)
    
tmp_lst = list(test_targets3)
final_org_vals = []
for i in range(len(tmp_lst)):
    temp = str(tmp_lst[i])
    temp_1 = str(temp).replace("TensorCategory(","")
    temp_2 = int(str(temp_1).replace(")",""))
    final_org_vals.append(temp_2)


dt_table = []
dt_return = fun_accrs(final_org_vals,final_pred_vals)
dt_table.append(dt_return)

dt_table_final = DataFrame(dt_table, columns=["Accuracy","Precision","Recall",
                                                    "F1 (weighted)","F1 (Macro)","F1 (Micro)","ROC AUC"])

print(dt_table_final)
########################## Evaluation Metrics (End) ###########################

################################ Time Series to RP (Ends) ###################################

print("All Processing Done!!!")

sys.stdout.close()



