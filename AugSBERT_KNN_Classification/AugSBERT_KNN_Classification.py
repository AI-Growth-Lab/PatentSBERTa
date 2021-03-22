#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.spatial
import numpy as np
import os, json
import glob
import re
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)
import pandas as pd
import torch
import random
import itertools

F1Measure_list = []
Recall_list = []
Accuracy_list = []
Precision_list = []
Hamming_Loss_list = []


def get_top_n_similar_patents_df(new_claim, claims):

    embedder = SentenceTransformer('/user/id.aau.dk/gy22ge/output/bi-encoder/stsb_augsbert_SS_roberta-base-2021-01-06_22-14-54')
    query_embeddings = embedder.encode([new_claim])
    claim_embeddings = embedder.encode(claims)
    top_n = 40
    distances = scipy.spatial.distance.cdist(query_embeddings, claim_embeddings, "cosine")[0]
    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])
    top_claim_ids = []
    top_claims = []
    top_similarity_scores = []

    # Find the closest 100 patent claims for each new_claim based on cosine similarity
    for idx, distance in results[0:top_n]:
        top_claim_ids.append(patent_id[idx])
        top_claims.append(claims[idx])
        top_similarity_scores.append(round((1-distance), 4))
        print('Patent ID: ' + str(patent_id[idx]))
        print('PubMed Claim: ' + claims[idx])
        print('Similarity Score: ' + "%.4f" % (1-distance))
        print('\n')
        
    top_100_similar_patents_df = pd.DataFrame({
        'top_claim_ids': top_claim_ids,
        'cosine_similarity': top_similarity_scores,
        'claims': top_claims,
    })
    
    return top_100_similar_patents_df

def F1Measure(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
            continue
        temp+= (2*sum(np.logical_and(y_true[i], y_pred[i])))/ (sum(y_true[i])+sum(y_pred[i]))
    return temp/ y_true.shape[0]

def Recall(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if sum(y_pred[i]) == 0:
            continue
        temp+= sum(np.logical_and(y_true[i], y_pred[i]))/ sum(y_pred[i])
    return temp/ y_true.shape[0]

def Precision(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        if sum(y_true[i]) == 0:
            continue
        temp+= sum(np.logical_and(y_true[i], y_pred[i]))/ sum(y_true[i])
    return temp/ y_true.shape[0]

def Hamming_Loss(y_true, y_pred):
    temp=0
    for i in range(y_true.shape[0]):
        temp += np.size(y_true[i] == y_pred[i]) - np.count_nonzero(y_true[i] == y_pred[i])
    return temp/(y_true.shape[0] * y_true.shape[1])

def Accuracy(y_true, y_pred):
    temp = 0
    for i in range(y_true.shape[0]):
        temp += sum(np.logical_and(y_true[i], y_pred[i])) / sum(np.logical_or(y_true[i], y_pred[i]))
    return temp / y_true.shape[0]

df_claim_cpc_test = pd.read_csv('/user/id.aau.dk/gy22ge/database/df_500_L_15723_test_100_20.csv', encoding='ISO-8859-1')
df_claim_cpc_train = pd.read_csv('/user/id.aau.dk/gy22ge/database/df_500_L_15723.csv', encoding='ISO-8859-1')
# df_claim_cpc_train_1000 = pd.read_csv('/home/ubuntu/deeppatentsimilarity/patentdata/df_claim_cpc_all_len_150_200_1000.csv', encoding='ISO-8859-1')
# df_claim_cpc_test = pd.read_csv('/home/ubuntu/deeppatentsimilarity/patentdata/prelabel/NewTest/df_1_L_43259_test_100.csv', encoding='ISO-8859-1')
# df_claim_cpc_train = pd.read_csv('/home/ubuntu/deeppatentsimilarity/patentdata/prelabel/NewTest/df_1_L_43259.csv', encoding='ISO-8859-1')

claims = list(df_claim_cpc_train.text)
patent_id = list(df_claim_cpc_train.patent_id)

listofpredictdfs = []

for i in range(len(df_claim_cpc_test)):
    get_top_n_similar_patents_df_predict = get_top_n_similar_patents_df(df_claim_cpc_test.text.iloc[i], claims)
    result = pd.merge(get_top_n_similar_patents_df_predict, df_claim_cpc_train, left_on='top_claim_ids',right_on='patent_id',how='left',suffixes=('_left','_right'))
    locals()["predict_n"+str(i)] = result.copy()
    listofpredictdfs.append("predict_n"+str(i))

df = pd.concat(map(lambda x: eval(x), listofpredictdfs),keys= listofpredictdfs ,axis=0)

top_k = 40
for k in range(top_k):
    top_n = k
    predict = pd.DataFrame(columns= df_claim_cpc_test.columns[11:])
    for item in range(len(listofpredictdfs)):
        k_similar_patents = df.xs(listofpredictdfs[item]).nlargest(top_n, ['cosine_similarity'])
        result_k_similar_patents = pd.DataFrame(0, index=np.arange(1),columns= k_similar_patents.columns[14:])
        for i in range(top_n):
            result_k_similar_patents  = result_k_similar_patents + k_similar_patents.iloc[i, 14:].values
            
        result_k_similar_patents_df = pd.DataFrame(result_k_similar_patents, columns= k_similar_patents.columns[14:])
        result_k_similar_patents_df.insert(0, "input_aptent_id", df_claim_cpc_test.patent_id.iloc[item], True)
        locals()["predict"+str(item)] = result_k_similar_patents_df.copy()
    
        predict = pd.concat([predict, locals()["predict"+str(item)]], ignore_index=True)
        result_k_similar_patents_df = result_k_similar_patents_df[0:0]
        
    predict_t = predict.iloc[:,:20]
    data = torch.tensor((predict_t.to_numpy()).astype(float), dtype=torch.float32)
    m = nn.Sigmoid()
    output = m(data)
    output = (output>0.9).float()
    output_df = pd.DataFrame(output, columns=predict.columns[:20]).astype(float)
    y_pred = output_df.to_numpy()
    y_true = df_claim_cpc_test.iloc[:, 11:31].to_numpy()
    F1Measure_list.append(F1Measure(y_true,y_pred))
    Recall_list.append(Recall(y_true,y_pred))
    Accuracy_list.append(Accuracy(y_true, y_pred))
    Precision_list.append(Precision(y_true,y_pred))
    Hamming_Loss_list.append(Hamming_Loss(y_true, y_pred))

    print("F1Measure: ", F1Measure(y_true,y_pred))
    print("Recall: ", Recall(y_true,y_pred))
    print("Accuracy: ", Accuracy(y_true, y_pred))
    print("Precision: ", Precision(y_true,y_pred))
    print("Hamming_Loss: ", Hamming_Loss(y_true, y_pred))

# dictionary of lists   
dict = {'F1Measure': F1Measure_list, 'Recall': Recall_list, 'Accuracy': Accuracy_list, 'Precision': Precision_list, 'Hamming_Loss': Hamming_Loss_list}   
       
df_ModelMetrics = pd.DataFrame(dict)  
    
# saving the dataframe  
df_ModelMetrics.to_csv('/user/id.aau.dk/gy22ge/output/modelpredict/Model_Metrics.csv'+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), index = False)
predict.to_csv(r'/user/id.aau.dk/gy22ge/output/modelpredict/predict_result_filteraugsbert.csv'+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), index = False)

