import pandas as pd
import numpy as np
import torch
from scipy.special import expit
import math

import torch
import torch.nn.functional as F
import torch.nn as nn
import random
import os
import itertools

import warnings
warnings.filterwarnings('ignore')

# def encoding(features_to_vary, N):
#     matrix = np.zeros((N-1, N-1))
#     # print("matrix = ", matrix)
#     for i in features_to_vary:
#         # print("matrix[",i,"-1][",i,"-1]=", matrix[i-1][i-1])
#         matrix[i-1][i-1] = 1
#         # print("matrix[",i,"-1][",i,"-1]=", matrix[i-1][i-1])
#     # print("matrix = ", matrix)
#     return matrix




# x=x[0]
# features_to_vary = [1,3,4]
# encoding_mat = encoding(features_to_vary, len(data.columns))
# print("data = \n", data)
# print("x = ", x)
# print("encoding_mat = \n",encoding_mat)
# print("x*encoding_mat = \n", np.dot(x,encoding_mat))
# print(type(encoding_mat))

# # matrix = np.random.randint(low=1, high=100, size = (N-1, N-1))
# # print(type(matrix))

# # print("matrix = \n", matrix)
# # for i in features_to_vary:
# #     print("matrix[",i,"-1][",i,"-1]=", matrix[i-1][i-1])
# #     matrix[i-1][i-1] = -1
# #     print("matrix[",i,"-1][",i,"-1]=", matrix[i-1][i-1])
# # print("matrix = \n", matrix)


# def encoding2(features_to_vary, N):
#     matrix = [0]*(N-1)
#     # print("matrix = ", matrix)
#     for i in features_to_vary:
#         # print("matrix[",i,"-1][",i,"-1]=", matrix[i-1][i-1])
#         matrix[i-1] = 1
#         # print("matrix[",i,"-1][",i,"-1]=", matrix[i-1][i-1])
#     # print("matrix = ", matrix)
#     return np.array(matrix)

# print(encoding2([1,3,4], 5))


# x = torch.Tensor([ -1.0000, -10.0000,  18.0189,  20.0000, -19.0000])
# print(x.numpy()



def actual_function1(inp1, inp2):
    val = 2*(inp1**2)+inp2*5 + 4
    if val>350:
        return 1
    return 0



def actual_function2(inp1, inp2):
    val = 1/(inp1**2+math.sqrt(abs(inp2))*3+1) + math.sin(3*inp1*inp2+5)
    if val >= 0.5:
        return 1
    return 0



def actual_function3(inp1, inp2, inp3, inp4, inp5):
    val = np.exp(-np.abs(inp1+inp2+inp3+inp4+inp5))+ np.random.normal(0, 1)
    if val>=0.5:
        return 1
    return 0



#func1
# print("Original output: ", actual_function1(5, -16))
# print("Final output: ", actual_function1(5. , 1.6682227))

# #func2
# print("Original output: ", actual_function2(-6,-14, 1, 3, -9))
# print("Final output: ", actual_function2(-6, -14, 1, 0.63676417, -9))

#func3
# print("Original output: ", (-6,-14, 1, 30, -9))
# print("Final output: ", actual_function3(-6, -14, 1, 30, -9))

# bce = nn.BCEWithLogitsLoss()
# bce2 = nn.BCELoss()
# print("\nBCE with logits loss:\n")
# print(bce(torch.as_tensor([float(0)]), torch.as_tensor([float(1)])))
# print(bce(torch.as_tensor([float(1)]), torch.as_tensor([float(0)])))
# print(bce(torch.as_tensor([float(0)]), torch.as_tensor([float(0.5)])))
# print(bce(torch.as_tensor([float(1)]), torch.as_tensor([float(0.5)])))
# print(bce(torch.as_tensor([float(0.5)]), torch.as_tensor([float(0.5)])))
# print(bce(torch.as_tensor([float(0)]), torch.as_tensor([float(0)])))
# print("\nBCE:\n")
# print(bce2(torch.as_tensor([float(0)]), torch.as_tensor([float(1)])))
# print(bce2(torch.as_tensor([float(1)]), torch.as_tensor([float(0)])))
# print(bce2(torch.as_tensor([float(0)]), torch.as_tensor([float(0.5)])))
# print(bce2(torch.as_tensor([float(1)]), torch.as_tensor([float(0.5)])))
# print(bce2(torch.as_tensor([float(0.5)]), torch.as_tensor([float(0.5)])))
# print(bce2(torch.as_tensor([float(0)]), torch.as_tensor([float(0)])))
# print("\nTorch:\n")
# print(torch.as_tensor([float(1)]))
# print(torch.as_tensor([float(1)])-torch.as_tensor([float(0)]))

def encoding(N):
    matrix = [(np.random.randint(low=1, high = 10))]*(N)
    return np.array(matrix)

# x = np.array([-6, -2, 12, 17, 7])
# N=x.shape[0]
# print(N)
# mask = torch.Tensor(encoding(N))
# print("mask = ", mask)
# x = torch.Tensor(x)
# print("x = ", x)
# x_cf = torch.rand(x.shape)*torch.max(x)
# print("x_cf = ", x_cf)
# print("1-mask = ", 1-mask)
# res1 = mask*x_cf + (1-mask)*x
# print("res1 = ", res1)
# res2 = torch.matmul(mask, x_cf) + torch.matmul(1-mask, x)
# print("res2 = ", res2)
# res3 = torch.add(torch.matmul(mask, x_cf), torch.matmul(1-mask, x))
# print("res3 = ", res3)

#Used to count number of inversions in array of differences arr (list)
def count_inversions(arr, n):
    temp_arr = [0]*n #temporary array to store the sorted elements
    return mergeSort(arr, temp_arr, 0, n-1)
 
#Use mergesort to count inversions
def mergeSort(arr, temp_arr, left, right):
    count = 0
    if left < right:
        mid = (left + right)//2
        count += mergeSort(arr, temp_arr,
                                left, mid)
        count += mergeSort(arr, temp_arr,
                                mid + 1, right) 
        count += merge(arr, temp_arr, left, mid, right)
    return count
 
#Merge function for mergesort 
def merge(arr, temp_arr, left, mid, right):
    i = left
    j = mid + 1
    k = left
    count = 0
    while i <= mid and j <= right:
        if arr[i] >= arr[j]:
            temp_arr[k] = arr[i]
            k += 1
            i += 1
        else:
            temp_arr[k] = arr[j]
            count += (mid-i + 1)
            k += 1
            j += 1
    while i <= mid:
        temp_arr[k] = arr[i]
        k += 1
        i += 1
    while j <= right:
        temp_arr[k] = arr[j]
        k += 1
        j += 1
    for loop_var in range(left, right + 1):
        arr[loop_var] = temp_arr[loop_var]
    return count


#Given: Input x (list), Generated Counterfactual x_cf (list), the number of features which were changed n (int), the list of features which were changed eligible_set_num (int)
#Output: The number of violations in x_cf
#NOTE: We simply dont do 1/abs(x-x_cf) since it does not respect the current features which are being changed, and will lead to 1/0
def calculate_violations(x, x_cf, n, eligible_set_num):
    diff = abs(x-x_cf) #only the features changed will be shown - if some feature x_i is supposed to be changed but is not changed, then we can still identify it based on eligible_set_num
    list_of_inverse_diff = []
    for i in eligible_set_num:
        # list_of_inverse_diff.append(1/(abs(x_cf[i-1]-x[i-1])))
        list_of_inverse_diff.append((abs(x_cf[i-1]-x[i-1])))
    print(list_of_inverse_diff)
    return count_inversions(list_of_inverse_diff, n)

#Given: Dataset data (dataframe), Number of features N (int)
#Returns list of elements where pref(i) > pref(j) if i<j
def get_preference_order(data, N, col_name):
    N=N-1
    w = [1/N]*N #used to store the user preference order from most preferred at lower index to least preferred in highest index in index values
    i=0
    print("order_user_num = ", w)
    suborders = sublists([i+1 for i in range(N)])
    # eps = [0]*len(suborders)
    print("suborders = ", suborders)
    print("length of suborders = ", len(suborders))
    print("length of suborders 2= ", N*(N-1)/2)
    k=0

    # for pair in suborders:
    #     ch = int(input("Enter which feature you prefer in terms of feature number - {} or {}: ".format(data.columns[pair[0]-1], data.columns[pair[1]-1])))
    #     prob = w[pair[0]-1]/(w[pair[0]-1] + w[pair[1]-1])
    #     # print("w[pair[0]-1] = ", w[pair[0]-1])
    #     # print("w[pair[1]-1] = ", w[pair[1]-1])
    #     # print("pair[0] = ", pair[0])
    #     # print("pair[1] = ", pair[1])
    #     # print("prob = ", prob)
    #     if prob>=0.5:
    #         if ch==pair[0]: #We predict f0>f1, and if that is agreed by the user, do this
    #             # w[pair[0]-1] = w[pair[0]-1]+np.random.uniform(0, 1/N**2)
    #             # w[pair[1]-1] = w[pair[1]-1]-np.random.uniform(0, 1/N**2)
    #             w[pair[0]-1] = w[pair[0]-1]+1/N**2
    #             w[pair[1]-1] = w[pair[1]-1]-1/N**2
    #         else: #We predict f0>f1, and if that is disagreed by the user, do this
    #             # w[pair[0]-1] = w[pair[0]-1]-np.random.uniform(0, 1/N**2)
    #             # w[pair[1]-1] = w[pair[1]-1]+np.random.uniform(0, 1/N**2)
    #             w[pair[0]-1] = w[pair[0]-1]-1/N**2
    #             w[pair[1]-1] = w[pair[1]-1]+1/N**2
    #     if prob<0.5:
    #         if ch==pair[0]: #We predict f0<f1, and if that is disagreed by the user, do this
    #             # w[pair[0]-1] = w[pair[0]-1]+np.random.uniform(0, 1/N**2)
    #             # w[pair[1]-1] = w[pair[1]-1]-np.random.uniform(0, 1/N**2)
    #             w[pair[0]-1] = w[pair[0]-1]+1/N**2
    #             w[pair[1]-1] = w[pair[1]-1]-1/N**2
    #         else: #We predict f0<f1, and if that is agreed by the user, do this
    #             # w[pair[0]-1] = w[pair[0]-1]-np.random.uniform(0, 1/N**2)
    #             # w[pair[1]-1] = w[pair[1]-1]+np.random.uniform(0, 1/N**2)
    #             w[pair[0]-1] = w[pair[0]-1]-1/N**2
    #             w[pair[1]-1] = w[pair[1]-1]+1/N**2
    #     # print("w on loop: w = ", w)
    # print("w = ", w)
    # order_user_num = [i[0]+1 for i in sorted(enumerate(w), key=lambda k: k[1], reverse=True)]
    # order_user = [data.columns[i-1] for i in order_user_num]
    # return order_user, order_user_num


def sublists(lst, index=0, current=[]):
    sub = []
    
    for subset in itertools.combinations(lst, 2):
        if subset:
            sub.append(list(subset))
    # for L in range(len(lst)+1):
    #     for subset in itertools.combinations(lst, L):
    #         if subset:
    #             sub.append(list(subset))
        
    return sub

# x = np.array([-6, -2, 12, 17, 7])
# x = torch.Tensor(x)
# x_cf = np.array([-6.3, -2.7, 13.5, 13.4, 7.3])
# x_cf = torch.Tensor(x_cf)
# y1 = x.detach().numpy()
# y2 = x_cf.detach().numpy()

# print(calculate_violations(y1, y2, 3, [5,3,4]))


# x = np.array([-1.07637191, -1.66685316, -0.68176796,  0.41582309, -0.69660804])
# y = np.array([-1.0763719,  -1.6668532,  -0.68176794,  1.3377824,   0.40903932])

# print(y-x)

#z-score normalisation
def normalize(dataset, col_name):
    for i in dataset.columns:
        if i!=col_name:
            dataset[i] = (dataset[i]-dataset[i].mean())/(dataset[i].std())
    return dataset

# dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/heloc.csv")
dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/dataset4.csv")
# dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/give_me_some_credit.csv")
data=pd.read_csv(dataset)
N=len(data.columns)
# col_name='RiskPerformance' #HELOC
col_name='Y' #dataset4
# col_name='SeriousDlqin2yrs' #give me some credit
# data = normalize(data, col_name)
data = normalize(data, col_name).drop([col_name], axis=1)
labels = [f'{col}' for i, col in enumerate(data.columns)]

# index_x=1
# if(index_x!=-1):
#     x = np.float32(np.reshape(data.drop([col_name], axis=1).iloc[index_x].tolist(), (1,N-1))[0]) #conversion to float32 is needed so that converting to Type32 tensor doesnt give unexpected errors
#     print('Chosen data point is {} at index {} with {} = {}.'.format(x, index_x, col_name, data.iloc[index_x][col_name]))
# else:
#     index_x = random.choice(data.index[data[col_name]==0].tolist())
#     x = np.float32(np.reshape(data.drop([col_name], axis=1).iloc[index_x].tolist(), (1,N-1))[0])
#     print('Chosen data point is {} at a randomized index {} with {} = {}.'.format(x, index_x, col_name, data.iloc[index_x][col_name]))



# get_preference_order(data, N, col_name)
# order_user, order_user_num = get_preference_order(data, N, col_name)
# print(order_user)














#----------------------------- causal ----------------------------



# import sys
# sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

# from scm.causallearn.search.FCMBased import lingam

# model = lingam.ICALiNGAM()
# # model = lingam.RCD()
# model.fit(data)

# from scm.causallearn.search.FCMBased.lingam.utils import make_dot

# make_dot(model.adjacency_matrix_, labels=labels).render('test.gv', view=True)

# import scm.graphviz as graphviz
# from scm.dowhy import CausalModel

# def make_graph(adjacency_matrix, labels=None):
#     idx = np.abs(adjacency_matrix) > 0.01
#     dirs = np.where(idx)
#     d = graphviz.Digraph(engine='dot')
#     names = labels if labels else [f'x{i}' for i in range(len(adjacency_matrix))]
#     for name in names:
#         d.node(name)
#     for to, from_, coef in zip(dirs[0], dirs[1], adjacency_matrix[idx]):
#         d.edge(names[from_], names[to], label=str(coef))
#     return d

# def str_to_dot(string):
#     '''
#     Converts input string from graphviz library to valid DOT graph format.
#     '''
#     graph = string.strip().replace('\n', ';').replace('\t','')
#     graph = graph[:9] + graph[10:-2] + graph[-1] # Removing unnecessary characters from string
#     return graph

# # Obtain valid dot format
# graph_dot = make_graph(model.adjacency_matrix_, labels=labels)

# # Define Causal Model
# model=CausalModel(
#         data = data,
#         treatment='X1',
#         outcome='X4',
#         graph=str_to_dot(graph_dot.source))




# # Identification
# identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
# # print(identified_estimand)

# # Estimation
# estimate = model.estimate_effect(identified_estimand,
#                                 method_name="backdoor.linear_regression",
#                                 control_value=5,
#                                 treatment_value=7.1,
#                                 confidence_intervals=True,
#                                 test_significance=True)
# print("Causal Estimate is " + str(estimate.value))


# # model._outcome = data.columns[1]
# model=CausalModel(
#         data = model._data,
#         treatment='X1',
#         outcome='X4',
#         graph=str_to_dot(graph_dot.source))


# # Identification
# identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
# # print(identified_estimand)

# # Estimation
# estimate = model.estimate_effect(identified_estimand,
#                                 method_name="backdoor.linear_regression",
#                                 control_value=3.5,
#                                 treatment_value=1,
#                                 confidence_intervals=True,
#                                 test_significance=True)
# print("Causal Estimate after change is " + str(estimate.value))






#-----------------------------------------------------------------------------------------------








# adj = model.adjacency_matrix_

# final_adj = []

# for i in adj:
#     x = []
#     for j in i:
#         if j:
#             x.append(1)
#         else:
#             x.append(0)
#     final_adj.append(x)

# print(adj)
# print(final_adj)

# for i, row in data.iterrows():
    

# print([data.columns[i-1] for i in model.causal_order_])

# from scm.causallearn.search.FCMBased.lingam.utils import make_dot

# make_dot(model.adjacency_matrix_, labels=labels).render('test.gv', view=True)

# import itertools
# features_to_vary = [5, 4, 3]
# list_of_permutations = [list(x) for x in list(itertools.permutations(features_to_vary))]
# print(list_of_permutations)


# list_of_other_ft = list(set(data.columns) - set([data.columns[1]]))
# print(set(data.columns))
# print(set([data.columns[1]]))
# print(list_of_other_ft)

# list_of_other_ft = list(set([t for t in range(1,N)]) - set([3])) #The other features
# print(list_of_other_ft)


# x = np.array([-6, -2, 12, 17, 7])
# x = torch.Tensor(x)
# print(x[0], type(x[0]))
# print(x[0].item(), type(x[0].item()))





# import sys
# sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../"))

# from Carla.carla import MLModelCatalog
# from Carla.carla.data.catalog import OnlineCatalog
# from Carla.carla.recourse_methods import GrowingSpheres
# from src.algorithms.FourthMethod import FourthMethod

# data_name = "adult"
# dataset = OnlineCatalog(data_name)

# model = MLModelCatalog(dataset, "ann", "pytorch")

# hyperparameters = {}
# gs = GrowingSpheres(model, hyperparameters)

# factuals = dataset.df.sample(10)
# counterfactuals = gs.get_counterfactuals(factuals)

# # print(type(counterfactuals))
# print(counterfactuals)




# # x = torch.rand(size=4, requires_grad=True)
# # y = torch.rand(size=4, requires_grad=True)
# x = torch.rand(4)
# y = torch.rand(4)
# t1 = torch.tensor(np.random.rand(), requires_grad=False)
# t2 = torch.tensor(np.random.rand(), requires_grad=True)
# t3 = torch.tensor(np.random.rand(), requires_grad=True)
# t4 = torch.tensor(np.random.rand(), requires_grad=False)
# print(t1)
# print(t2)
# print(t3)
# print(t4)


# t = torch.cat((t1.reshape(1), t2.reshape(1), t3.reshape(1), t4.reshape(1)), dim=0)
# x.requires_grad = True
# y.requires_grad = True
# print(x)
# print(y)
# print(t)
# print("----reshape----")
# with torch.no_grad():
#     x=torch.reshape(x, (-1,1))
#     y=torch.reshape(y, (-1,1))


# print(x)
# print(y)
# x[0].requires_grad = False
# x[1].requires_grad = True
# x[2].requires_grad = True
# optimizer = torch.optim.SGD([x[1], x[2]], 0.01)

# for i in range(2):
#     print("-------------------Iteration ", i, "-------------------")
#     z= x+y
#     print(z)
    
#     loss = torch.exp(z).sum()
#     loss.backward()
#     print("----backward----")
#     optimizer.step()
#     print(x)
#     print("x.grad = ", x.grad)
#     print(y)
#     print("y.grad = ", y.grad)
#     print(z)
#     print("z.grad = ", z.grad)
#     print(loss)
#     print("loss.grad = ", loss.grad)



filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential3_model_D11.pt")
model = torch.jit.load(filename)
model.eval() #no more training on this model

dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/heloc.csv")
size_of_dataset = 200000
data=pd.read_csv(dataset)
dataset_name = os.path.splitext(os.path.basename(dataset))[0] #extract the name of the dataset used from the path given in dataset (no extension)
#output column name
if(dataset_name[:-1]=="dataset"):
    col_name = 'Y'
else:
    if(dataset_name=="adult"):
        #categorical_encoding
        col_name = 'income'
    elif(dataset_name=="compas"):
        #categorical_encoding
        col_name = 'score'
    elif(dataset_name=="give_me_some_credit"):
        col_name = 'SeriousDlqin2yrs'
    elif(dataset_name=="heloc"):
        col_name = 'RiskPerformance'

l1 = data.index[data[col_name]==1].tolist()
l2 = data.index[data[col_name]==0].tolist()
print("Number of ones = ", len(l1))
print("Number of zeroes = ", len(l2))

if(size_of_dataset>data.shape[0]):
    size_of_dataset = data.shape[0]
data = data.head(size_of_dataset) #size of dataset to be used


N=len(data.columns)
normalization="zscore"

y_hat = torch.as_tensor([float(1)])
p = 0

# index_x = 9
while y_hat.item()==1 and p<size_of_dataset:
    print("--------------------------------p = ", p, "--------------------------------")
    index_x = random.choice(data.index[data[col_name]==0].tolist())
    print("index_x = ", index_x, "and x = ", np.float32(np.reshape(data.drop([col_name], axis=1).iloc[index_x].tolist(), (1,N-1))[0]), " with y = ", data.iloc[index_x][col_name])
    t = torch.tensor(np.float32(np.reshape(data.drop([col_name], axis=1).iloc[index_x].tolist(), (1,N-1))[0]), dtype=torch.float32)
    print("t = ", t)
    y_hat = model(t)
    print("y_hat = ", y_hat.item())
    p+= 1
    
#save the normalized data to check
data.to_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/normalized", dataset_name + "_normalized.csv"), index=False)

#If x given by user
if(index_x!=-1):
    x = np.float32(np.reshape(data.drop([col_name], axis=1).iloc[index_x].tolist(), (1,N-1))[0]) #conversion to float32 is needed so that converting to Type32 tensor doesnt give unexpected errors
#If x is not given by user
else:
    index_x = random.choice(data.index[data[col_name]==0].tolist())
    x = np.float32(np.reshape(data.drop([col_name], axis=1).iloc[index_x].tolist(), (1,N-1))[0])

print(x, " with y = ", data.iloc[index_x][col_name])