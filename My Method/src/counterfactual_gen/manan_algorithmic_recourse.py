"""
This is an implementation of Manan's base work with no causality.

"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
import pandas as pd
import numpy as np
# np.set_printoptions(precision=3) #only 3 decimal digits will be shown for any numpy array
from dataset_gen import dataset_generator as dg
import joblib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../"))
from algorithms import WachterCF
import datetime
import copy
import random
from time import time
# from counterfactual_gen import Custom_Model


import logging
log_file_name = "Log_manan_algorithmic_recourse.log"
log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../logs", log_file_name)
logger = logging.getLogger()
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(log_path, mode='w')
fh.setLevel(logging.INFO)
# fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

import warnings
warnings.filterwarnings('ignore')


import torch
import torch.nn as nn
import torch.nn.functional as F


#Given: Dataset data (dataframe), Number of features N (int)
#Returns list of elements where pref(i) > pref(j) if i<j
def get_preference_order(data, N, col_name):
    order_user = ['']*N #used to store the user preference order from most preferred at lower index to least preferred in highest index in feature names
    order_user_num = ['']*N #used to store the user preference order from most preferred at lower index to least preferred in highest index in index values
    i=0
    print("Ordering indices start from 1. For example, feature with order index 1 means it is highest priority, 2 is lesser priority,  and so on.")
    for col in data.columns:
        if(col!=col_name):
            print("Feature "+str(i+1)+": "+col)
            i=i+1
    print("Enter in order the preference below:")
    for i in range(1, N):
        inp =int(input("Enter feature of preference " + str(i)+" (only the feature number): "))
        order_user[i-1] = data.columns[inp-1] 
        order_user_num[i-1] = inp #used to store the user preference order in the original order of the dataframe

    return order_user, order_user_num

#returns the model loaded from pkl files
def get_model(model_name):
    model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", model_name)
    return joblib.load(model_save_path)



#Given: List of features ft (list), dataset data (dataframe), model (model), input x (list)
#Output: Counterfactual using all the features in ft
def gen_cf(gap_between_prints, ft, data, N, model, lr, max_iter, l, x):
    obj = WachterCF.WachterCF(data, model)
    x_cf = obj.generate_counterfactuals(x, features_to_vary = ft, N=N, learning_rate=lr, max_iter=max_iter, _lambda=l, target = 1, gap_between_prints = gap_between_prints) #to be implemented
    return x_cf




#Used to count number of inversions in array of differences arr (list)
def count_inversions(arr, n):
    temp_arr = [0]*n #temporary array to store the sorted elements
    return func_count(arr, temp_arr, 0, n-1)
 
#Use mergesort to count inversions
def func_count(arr, temp_arr, left, right):
    count = 0
    if left < right:
        mid = (left + right)//2
        count += func_count(arr, temp_arr,
                                left, mid)
        count += func_count(arr, temp_arr,
                                mid + 1, right) 
        count += func_merge(arr, temp_arr, left, mid, right)
    return count
 
#Merge function for mergesort 
def func_merge(arr, temp_arr, left, mid, right):
    i = left
    j = mid + 1
    k = left
    count = 0
    while i <= mid and j <= right:
        if arr[i] >= arr[j]: #Here we count only if a[i]<a[j] if i<j i.e., the change in feature higher up in preference should be MORE
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




#Given: Input x (Numpy Array), Generated Counterfactual x_cf (Numpy Array)
#Output: The number of changes in x_cf compared to x
def find_change_count(x, x_counterfactual):
    diff = x_counterfactual-x
    count = 0
    for i in diff:
        if i!=0:
            count += 1
    return count




#Given: Input x (Numpy Array), Generated Counterfactual x_cf (Numpy Array)
#Output: The L1 distance between x and x_cf
def find_distance(x, x_counterfactual):
    diff = abs(x_counterfactual-x)
    distance = 0
    for i in diff:
        distance += i
    return distance




#Given: Input x (Numpy Array), Generated Counterfactual x_cf (Numpy Array), the number of features which were changed n (int), the list of features which were changed eligible_set_num (int)
#Output: The number of violations in x_cf
def calculate_violations(x, x_cf, n, eligible_set_num):
    list_of_diff = []
    for i in eligible_set_num:
        if i !='':
            list_of_diff.append(abs(x_cf[i-1]-x[i-1]))
    return count_inversions(list_of_diff, n)

# #Given: Input x (list), Generated Counterfactual x_cf (list), the number of features which were changed n (int), the list of features which were changed eligible_set_num (int)
# #Output: The number of violations in x_cf
# def calculate_violations(x, x_cf, n, eligible_set_num):
#     # diff = abs(x_cf-x) #only the features changed will be shown - if some feature x_i is supposed to be changed but is not changed, then we can still identify it based on eligible_set_num
#     list_of_diff = []
#     inversions = 0
#     for i in eligible_set_num:    
#         list_of_diff.append(abs(x_cf[i-1]-x[i-1]))
#     # for i in range(n-1):
#     #     if list_of_diff[i]<list_of_diff[i+1]:
#     #         inversions += 1
#     #         list_of_diff[i+1] = list_of_diff[i] #making it equal so that further down the line we can keep tracking if there is another which breaks order
#     return count_inversions(list_of_diff, n)
#     # return inversions
    



#Given: Counterfactual x_cf, Name of dataset dataset_name
#Output: Actual value of y
def find_actual_value(x_cf, dataset_name):
    if(dataset_name=='dataset1'):
        inp1 = x_cf[0]
        inp2 = x_cf[1]
        y_act = dg.actual_function1(inp1, inp2)
    elif(dataset_name=='dataset2'):
        inp1 = x_cf[0]
        inp2 = x_cf[1]
        y_act = dg.actual_function2(inp1, inp2)
    elif(dataset_name=='dataset3'):
        inp1 = x_cf[0]
        inp2 = x_cf[1]
        inp3 = x_cf[2]
        inp4 = x_cf[3]
        inp5 = x_cf[4]
        y_act = dg.actual_function3(inp1, inp2, inp3, inp4, inp5)
    elif(dataset_name=='dataset4'):
        inp1 = x_cf[0]
        inp2 = x_cf[1]
        inp3 = x_cf[2]
        inp4 = x_cf[3]
        inp5 = x_cf[4]
        y_act = dg.actual_function7(inp1, inp2, inp3, inp4, inp5)
    elif(dataset_name=='dataset5'):
        inp1 = x_cf[0]
        inp2 = x_cf[1]
        y_act = dg.actual_function5(inp1, inp2)
    elif(dataset_name=='dataset6'):
        inp1 = x_cf[0]
        inp2 = x_cf[1]
        inp3 = x_cf[2]
        inp4 = x_cf[3]
        inp5 = x_cf[4]
        y_act = dg.actual_function6(inp1, inp2, inp3, inp4, inp5)
    elif(dataset_name=='dataset7'):
        inp1 = x_cf[0]
        inp2 = x_cf[1]
        inp3 = x_cf[2]
        inp4 = x_cf[3]
        inp5 = x_cf[4]
        y_act = dg.actual_function7(inp1, inp2, inp3, inp4, inp5)
    else:
        y_act = -1
    
    return y_act


# #min max normalization
# def normalize(dataset, col_name):
#     for i in dataset.columns:
#         if i!=col_name:
#             dataset[i] = (dataset[i]-dataset[i].min())/(dataset[i].max()-dataset[i].min())
#     return dataset

#z-score normalisation
def normalize(dataset, col_name):
    for i in dataset.columns:
        if i!=col_name:
            dataset[i] = (dataset[i]-dataset[i].mean())/(dataset[i].std())
    return dataset

# #just feature scaling
# def normalize(dataset, col_name):
#     for i in dataset.columns:
#         if i!=col_name:
#             dataset[i] = dataset[i]/1000
#     return dataset



#Given: Preference order pref_order (list), dataset df (dataframe), number of features N (int), model (model), input x (list)
#Output: Counterfactual which causes least number of violations
def run(gap_between_prints, dataset, order_user, order_user_num, data, N, model, lr, l, max_iter, x):
    max_violation = 0
    avg_violation = 0
    min_violation = float('inf')
    best_counterfactual = []
    time_taken = 0.0
    logger.debug("{}: order_user_num = {}".format(datetime.datetime.now(), order_user_num))
    dataset_name = os.path.splitext(os.path.basename(dataset))[0] #extract the name of the dataset used from the path given in dataset (no extension)
    for i in range(N-1):
        logger.debug("{}: i = {}".format(datetime.datetime.now(), i))
        eligible_set = order_user[0:i+1]
        eligible_set_num = order_user_num[0:i+1]
        logger.info("{}: Current features = {}".format(datetime.datetime.now(), eligible_set_num))
        time_start = time()
        x_counterfactual = gen_cf(gap_between_prints, eligible_set_num, data, N, model, lr, max_iter, l, x).detach().numpy()
        time_end = time()
        time_taken += time_end-time_start
        logger.debug("{}: Time taken for this set is = {} seconds".format(datetime.datetime.now(), time_end-time_start))
        violations = calculate_violations(x, x_counterfactual, i+1, eligible_set_num)
        # violations = 10000
        if(violations>max_violation):
            max_violation=violations
        if(violations<min_violation):
            min_violation=violations
            best_counterfactual=copy.deepcopy(x_counterfactual)
        elif(violations==min_violation and find_distance(x, x_counterfactual)<find_distance(x, best_counterfactual)): #First priority is number of violations, followed by L1 distance
            best_counterfactual=copy.deepcopy(x_counterfactual)
        
        avg_violation = avg_violation + violations
        y_act = find_actual_value(x_counterfactual, dataset_name)
        if y_act!=-1:
            logger.debug("{}: Actual y for this counterfactual is = {}".format(datetime.datetime.now(), y_act))
        else:
            logger.debug("{}: Actual output for this counterfactual is not possible since this is a real world dataset.".format(datetime.datetime.now()))
        logger.debug("-------------------------------------------------------------------------------------------------------\n")
    avg_violation = avg_violation/N
    # time_taken = time_taken/N
    y_act = find_actual_value(best_counterfactual, dataset_name)
    if y_act!=-1:
        logger.info("{}: Actual y for this counterfactual is = {}".format(datetime.datetime.now(), y_act))
    else:
        logger.info("{}: Actual output for this counterfactual is not possible since this is a real world dataset.".format(datetime.datetime.now()))
    logger.info("-------------------------------------------------------------------------------------------------------\n")
    return best_counterfactual, max_violation, min_violation, avg_violation, time_taken
    # return x_counterfactual, max_violation, min_violation, avg_violation
    


#Given: The absolute path of the dataset to be used dataset (string), Index of preferred datapoint to find counterfactual of x (int - default value = -1)
#Output: Counterfactual which causes least number of violations
def main(dataset, index_x=-1,  lr=0.01, l=0.1, max_iter=100, gap_between_prints=80, size_of_dataset=5000, seed=1, type_of_model = -1, order_user = None, order_user_num = None):
    #Input section
    data=pd.read_csv(dataset)
    if(size_of_dataset>data.shape[0]):
        size_of_dataset = data.shape[0]
    data = data.head(size_of_dataset) #size of dataset to be used
    dataset_name = os.path.splitext(os.path.basename(dataset))[0] #extract the name of the dataset used from the path given in dataset (no extension)
    N=len(data.columns)
    
    # if(index_x!=-1):
    #     x = np.reshape(data.drop(['Y'], axis=1).iloc[index_x].tolist(), (1,N-1))[0]
    #     logger.debug('{}: Chosen data point is {} at index {}.'.format(datetime.datetime.now(), x, index_x))
    # else:
    #     # logger.debug('{}: Chosen data point is {} at index {}.'.format(datetime.datetime.now(), data.loc[data["Y"]==0].sample().values[0], index_x))
    #     # x = data.loc[data["Y"]==0].sample().drop(['Y'], axis=1).values[0]
    #     index_x = data.index(data.loc[data["Y"]==0].sample())
    #     logger.debug('{}: Chosen data point is {} at a randomized index.'.format(datetime.datetime.now(), x))
    
    #output column name
    if(dataset_name[:-1]=="dataset"):
        col_name = 'Y'
    else:
        if(dataset_name=="adult"):
            col_name = 'income'
        elif(dataset_name=="compas"):
            #categorical_encoding
            col_name = 'score'
        elif(dataset_name=="give_me_some_credit"):
            col_name = 'SeriousDlqin2yrs'
        elif(dataset_name=="heloc"):
            col_name = 'RiskPerformance'
    
    data = normalize(data, col_name)

    if(index_x!=-1):
        x = np.float32(np.reshape(data.drop([col_name], axis=1).iloc[index_x].tolist(), (1,N-1))[0]) #conversion to float32 is needed so that converting to Type32 tensor doesnt give unexpected errors
        logger.info('{}: Chosen data point is {} at index {} with {} = {}.'.format(datetime.datetime.now(), x, index_x, col_name, data.iloc[index_x][col_name]))
    else:
        # logger.debug('{}: Chosen data point is {} at index {}.'.format(datetime.datetime.now(), data.loc[data["Y"]==0].sample().values[0], index_x))
        # x = data.loc[data["Y"]==0].sample().drop(['Y'], axis=1).values[0]
        index_x = random.choice(data.index[data[col_name]==0].tolist())
        x = np.float32(np.reshape(data.drop([col_name], axis=1).iloc[index_x].tolist(), (1,N-1))[0])
        logger.info('{}: Chosen data point is {} at a randomized index {} with {} = {}.'.format(datetime.datetime.now(), x, index_x, col_name, data.iloc[index_x][col_name]))
    
    if order_user == None or order_user_num == None:
        order_user, order_user_num = get_preference_order(data, N, col_name)

    if type_of_model==-1:
        ch = int(input("""Choose the type of Model:
                    \n1) Custom Sequential 1
                    \n2) Custom Sequential 2
                    \n3) Custom Sequential 3
                    \n4) MLP Classifier
                    \nChoose the number: """))
    else:
        ch = type_of_model
    if(ch==1):
        if(dataset_name=='dataset1'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential1_model_D1.pt")
        elif(dataset_name=='dataset2'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential1_model_D2.pt")
        elif(dataset_name=='dataset3'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential1_model_D3.pt")
        if(dataset_name=='dataset4'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential1_model_D4.pt")
        elif(dataset_name=='dataset5'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential1_model_D5.pt")
        elif(dataset_name=='dataset6'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential1_model_D6.pt")
        elif(dataset_name=='dataset7'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential1_model_D7.pt")
        elif(dataset_name=='adult'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential1_model_D8.pt")
        elif(dataset_name=='compas'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential1_model_D9.pt")
        elif(dataset_name=='give_me_some_credit'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential1_model_D10.pt")
        elif(dataset_name=='heloc'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential1_model_D11.pt")
    elif(ch==2):
        if(dataset_name=='dataset1'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential2_model_D1.pt")
        elif(dataset_name=='dataset2'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential2_model_D2.pt")
        elif(dataset_name=='dataset3'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential2_model_D3.pt")
        if(dataset_name=='dataset4'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential2_model_D4.pt")
        elif(dataset_name=='dataset5'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential2_model_D5.pt")
        elif(dataset_name=='dataset6'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential2_model_D6.pt")
        elif(dataset_name=='dataset7'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential2_model_D7.pt")
        elif(dataset_name=='adult'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential2_model_D8.pt")
        elif(dataset_name=='compas'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential2_model_D9.pt")
        elif(dataset_name=='give_me_some_credit'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential2_model_D10.pt")
        elif(dataset_name=='heloc'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential2_model_D11.pt")
    elif(ch==3):
        if(dataset_name=='dataset1'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential3_model_D1.pt")
        elif(dataset_name=='dataset2'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential3_model_D2.pt")
        elif(dataset_name=='dataset3'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential3_model_D3.pt")
        if(dataset_name=='dataset4'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential3_model_D4.pt")
        elif(dataset_name=='dataset5'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential3_model_D5.pt")
        elif(dataset_name=='dataset6'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential3_model_D6.pt")
        elif(dataset_name=='dataset7'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential3_model_D7.pt")
        elif(dataset_name=='adult'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential3_model_D8.pt")
        elif(dataset_name=='compas'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential3_model_D9.pt")
        elif(dataset_name=='give_me_some_credit'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential3_model_D10.pt")
        elif(dataset_name=='heloc'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential3_model_D11.pt")
    elif(ch==4): #default is taken as MLP model
        if(dataset_name=='dataset1'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/mlp_classifier_model_D1.pt")
        elif(dataset_name=='dataset2'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/mlp_classifier_model_D2.pt")
        elif(dataset_name=='dataset3'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/mlp_classifier_model_D3.pt")
        if(dataset_name=='dataset4'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/mlp_classifier_model_D4.pt")
        elif(dataset_name=='dataset5'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/mlp_classifier_model_D5.pt")
        elif(dataset_name=='dataset6'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/mlp_classifier_model_D6.pt")
        elif(dataset_name=='dataset7'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/mlp_classifier_model_D7.pt")
        elif(dataset_name=='adult'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/mlp_classifier_model_D8.pt")
        elif(dataset_name=='compas'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/mlp_classifier_model_D9.pt")
        elif(dataset_name=='give_me_some_credit'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/mlp_classifier_model_D10.pt")
        elif(dataset_name=='heloc'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/mlp_classifier_model_D11.pt")
        
    
    model = torch.jit.load(filename)
    model.eval() #no more training on this model
    logger.debug("{}: order_user_num in main = {}".format(datetime.datetime.now(), order_user_num))
    #Output section
    best_counterfactual, max_violation, min_violation, avg_violation, time_taken = run(gap_between_prints, dataset, order_user, order_user_num, data, N, model,  lr, l, max_iter, x)
    
    if(model(torch.from_numpy(best_counterfactual))):
        number_of_features_changed = find_change_count(x, best_counterfactual)
        cost_of_counterfactual = find_distance(x, best_counterfactual)
        logger.info('{}: Original data point is {}'.format(datetime.datetime.now(), x))
        logger.info('{}: The counterfactual is given by {} with number of vioaltions = {}, number of features changed = {}, and L1 cost = {}'.format(datetime.datetime.now(), best_counterfactual, min_violation, number_of_features_changed, cost_of_counterfactual))
        logger.info('{}: Maximum number of violations is {}'.format(datetime.datetime.now(), max_violation))
        logger.info('{}: Average number of violations is {}'.format(datetime.datetime.now(), avg_violation))
        logger.info('{}: Average Time taken {} seconds'.format(datetime.datetime.now(), time_taken))
        # logger.info('{}: Model output on Counterfactual is {}'.format(datetime.datetime.now(), avg_violation))
    else:
        logger.info('{}: Could not find a counterfactual in the given iterations.'.format(datetime.datetime.now()))
        x_counterfactual = None
        number_of_features_changed = -1
        cost_of_counterfactual = -1

    return avg_violation, number_of_features_changed, cost_of_counterfactual, time_taken




#Given: The absolute path of the dataset to be used dataset (string), Index of preferred datapoint to find counterfactual of x (int - default value = -1)
#Output: Counterfactual which causes least number of violations
def main_automate(dataset, data, x, index_x=-1,  lr=0.01, l=0.1, max_iter=100, gap_between_prints=80, size_of_dataset=5000, seed=1, type_of_model = -1, order_user = None, order_user_num = None):
    #Input section
    if data is None:
        data=pd.read_csv(dataset)
        if(size_of_dataset>data.shape[0]):
            size_of_dataset = data.shape[0]
        data = data.head(size_of_dataset) #size of dataset to be used
        data = normalize(data, col_name)


    dataset_name = os.path.splitext(os.path.basename(dataset))[0] #extract the name of the dataset used from the path given in dataset (no extension)
    N=len(data.columns)
    
    # if(index_x!=-1):
    #     x = np.reshape(data.drop(['Y'], axis=1).iloc[index_x].tolist(), (1,N-1))[0]
    #     logger.debug('{}: Chosen data point is {} at index {}.'.format(datetime.datetime.now(), x, index_x))
    # else:
    #     # logger.debug('{}: Chosen data point is {} at index {}.'.format(datetime.datetime.now(), data.loc[data["Y"]==0].sample().values[0], index_x))
    #     # x = data.loc[data["Y"]==0].sample().drop(['Y'], axis=1).values[0]
    #     index_x = data.index(data.loc[data["Y"]==0].sample())
    #     logger.debug('{}: Chosen data point is {} at a randomized index.'.format(datetime.datetime.now(), x))
    
    #output column name
    if(dataset_name[:-1]=="dataset"):
        col_name = 'Y'
    else:
        if(dataset_name=="adult"):
            col_name = 'income'
        elif(dataset_name=="compas"):
            #categorical_encoding
            col_name = 'score'
        elif(dataset_name=="give_me_some_credit"):
            col_name = 'SeriousDlqin2yrs'
        elif(dataset_name=="heloc"):
            col_name = 'RiskPerformance'
    
    
    if x is None:
        if(index_x!=-1):
            x = np.float32(np.reshape(data.drop([col_name], axis=1).iloc[index_x].tolist(), (1,N-1))[0]) #conversion to float32 is needed so that converting to Type32 tensor doesnt give unexpected errors
            logger.info('{}: Chosen data point is {} at index {} with {} = {}.'.format(datetime.datetime.now(), x, index_x, col_name, data.iloc[index_x][col_name]))
        else:
            # logger.debug('{}: Chosen data point is {} at index {}.'.format(datetime.datetime.now(), data.loc[data["Y"]==0].sample().values[0], index_x))
            # x = data.loc[data["Y"]==0].sample().drop(['Y'], axis=1).values[0]
            index_x = random.choice(data.index[data[col_name]==0].tolist())
            x = np.float32(np.reshape(data.drop([col_name], axis=1).iloc[index_x].tolist(), (1,N-1))[0])
            logger.info('{}: Chosen data point is {} at a randomized index {} with {} = {}.'.format(datetime.datetime.now(), x, index_x, col_name, data.iloc[index_x][col_name]))
        
    if order_user is None or order_user_num is None:
        order_user, order_user_num = get_preference_order(data, N, col_name)

    if type_of_model==-1:
        ch = int(input("""Choose the type of Model:
                    \n1) Custom Sequential 1
                    \n2) Custom Sequential 2
                    \n3) Custom Sequential 3
                    \n4) MLP Classifier
                    \nChoose the number: """))
    else:
        ch = type_of_model
    if(ch==1):
        if(dataset_name=='dataset1'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential1_model_D1.pt")
        elif(dataset_name=='dataset2'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential1_model_D2.pt")
        elif(dataset_name=='dataset3'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential1_model_D3.pt")
        if(dataset_name=='dataset4'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential1_model_D4.pt")
        elif(dataset_name=='dataset5'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential1_model_D5.pt")
        elif(dataset_name=='dataset6'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential1_model_D6.pt")
        elif(dataset_name=='dataset7'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential1_model_D7.pt")
        elif(dataset_name=='adult'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential1_model_D8.pt")
        elif(dataset_name=='compas'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential1_model_D9.pt")
        elif(dataset_name=='give_me_some_credit'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential1_model_D10.pt")
        elif(dataset_name=='heloc'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential1_model_D11.pt")
    elif(ch==2):
        if(dataset_name=='dataset1'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential2_model_D1.pt")
        elif(dataset_name=='dataset2'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential2_model_D2.pt")
        elif(dataset_name=='dataset3'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential2_model_D3.pt")
        if(dataset_name=='dataset4'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential2_model_D4.pt")
        elif(dataset_name=='dataset5'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential2_model_D5.pt")
        elif(dataset_name=='dataset6'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential2_model_D6.pt")
        elif(dataset_name=='dataset7'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential2_model_D7.pt")
        elif(dataset_name=='adult'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential2_model_D8.pt")
        elif(dataset_name=='compas'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential2_model_D9.pt")
        elif(dataset_name=='give_me_some_credit'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential2_model_D10.pt")
        elif(dataset_name=='heloc'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential2_model_D11.pt")
    elif(ch==3):
        if(dataset_name=='dataset1'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential3_model_D1.pt")
        elif(dataset_name=='dataset2'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential3_model_D2.pt")
        elif(dataset_name=='dataset3'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential3_model_D3.pt")
        if(dataset_name=='dataset4'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential3_model_D4.pt")
        elif(dataset_name=='dataset5'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential3_model_D5.pt")
        elif(dataset_name=='dataset6'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential3_model_D6.pt")
        elif(dataset_name=='dataset7'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential3_model_D7.pt")
        elif(dataset_name=='adult'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential3_model_D8.pt")
        elif(dataset_name=='compas'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential3_model_D9.pt")
        elif(dataset_name=='give_me_some_credit'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential3_model_D10.pt")
        elif(dataset_name=='heloc'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential3_model_D11.pt")
    elif(ch==4): #default is taken as MLP model
        if(dataset_name=='dataset1'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/mlp_classifier_model_D1.pt")
        elif(dataset_name=='dataset2'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/mlp_classifier_model_D2.pt")
        elif(dataset_name=='dataset3'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/mlp_classifier_model_D3.pt")
        if(dataset_name=='dataset4'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/mlp_classifier_model_D4.pt")
        elif(dataset_name=='dataset5'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/mlp_classifier_model_D5.pt")
        elif(dataset_name=='dataset6'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/mlp_classifier_model_D6.pt")
        elif(dataset_name=='dataset7'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/mlp_classifier_model_D7.pt")
        elif(dataset_name=='adult'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/mlp_classifier_model_D8.pt")
        elif(dataset_name=='compas'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/mlp_classifier_model_D9.pt")
        elif(dataset_name=='give_me_some_credit'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/mlp_classifier_model_D10.pt")
        elif(dataset_name=='heloc'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/mlp_classifier_model_D11.pt")
        
    
    model = torch.jit.load(filename)
    model.eval() #no more training on this model
    logger.debug("{}: order_user_num in main = {}".format(datetime.datetime.now(), order_user_num))
    #Output section
    best_counterfactual, max_violation, min_violation, avg_violation, time_taken = run(gap_between_prints, dataset, order_user, order_user_num, data, N, model,  lr, l, max_iter, x)
    
    if(model(torch.from_numpy(best_counterfactual))):
        number_of_features_changed = find_change_count(x, best_counterfactual)
        cost_of_counterfactual = find_distance(x, best_counterfactual)
        logger.info('{}: Original data point is {}'.format(datetime.datetime.now(), x))
        logger.info('{}: The counterfactual is given by {} with number of vioaltions = {}, number of features changed = {}, and L1 cost = {}'.format(datetime.datetime.now(), best_counterfactual, min_violation, number_of_features_changed, cost_of_counterfactual))
        logger.info('{}: Maximum number of violations is {}'.format(datetime.datetime.now(), max_violation))
        logger.info('{}: Average number of violations is {}'.format(datetime.datetime.now(), avg_violation))
        logger.info('{}: Average Time taken {} seconds'.format(datetime.datetime.now(), time_taken))
        # logger.info('{}: Model output on Counterfactual is {}'.format(datetime.datetime.now(), avg_violation))
    else:
        logger.info('{}: Could not find a counterfactual in the given iterations.'.format(datetime.datetime.now()))
        x_counterfactual = None
        number_of_features_changed = -1
        cost_of_counterfactual = -1

    return avg_violation, number_of_features_changed, cost_of_counterfactual, time_taken





if __name__=="__main__":
    ch = int(input("""\n
                   Choose the Dataset:\n
                   1) Dataset 1\n
                   2) Dataset 2\n
                   3) Dataset 3\n
                   4) Dataset 4\n
                   5) Dataset 5\n
                   6) Dataset 6\n
                   7) Dataset 7\n
                   8) Adult\n
                   9) COMPAS\n
                   10) Give Me Some Credit\n
                   11) HELOC\n
                   Choose the number: """))
    if(ch==1):
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/dataset1.csv")
    elif(ch==2):
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/dataset2.csv")
    elif(ch==3): #default is taken as dataset 3
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/dataset3.csv")
    elif(ch==4):
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/dataset4.csv")
    elif(ch==5): #default is taken as dataset 3
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/dataset5.csv")
    elif(ch==6):
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/dataset6.csv")
    elif(ch==7): #default is taken as dataset 3
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/dataset7.csv")
    elif(ch==8): #default is taken as dataset 3
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/adult.csv")
    elif(ch==9): #default is taken as dataset 3
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/compas.csv")
    elif(ch==10): #default is taken as dataset 3
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/give_me_some_credit.csv")
    elif(ch==11): #default is taken as dataset 3
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/heloc.csv")
    # index_x = int(input("Choose the index of datapoint you want to change (-1 if you want random) :"))
    # learning_rate=float(input("Choose the value of learning rate (0.01 if you want basic): "))
    # _lambda = float(input("Choose the value of lambda (0.1 if you want basic): "))
    # max_iterations = int(input("Choose the number of iterations to run for (100 if you want basic): "))
        
    index_x = 90 #1 for HELOC, 2 for dataset3
    learning_rate = 1e-3
    _lambda = 0.7
    max_iterations = 10
    gap_between_prints = 1
    size_of_dataset = 5000
    seed = 3
    main(dataset,index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed)


