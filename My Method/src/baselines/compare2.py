import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
import pandas as pd
from counterfactual_gen import  manan_algorithmic_recourse2 as mananbase, manan_with_causality_algorithmic_recourse2 as manancausal, my_method_algorithmic_recourse as mymethod1, my_method_algorithmic_recourse3 as mymethod3, Algorithm5 as mymethod5, Algorithm8 as mymethod8, Algorithm10 as mymethod10
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../"))
import datetime
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from time import time


import logging
exp_name = "comparison2"
log_file_name = "Log_"+exp_name+".log"
log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../logs", log_file_name)
logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
fh = logging.FileHandler(log_path, mode='w')
# fh.setLevel(logging.DEBUG)
fh.setLevel(logging.INFO)
logger.addHandler(fh)

import warnings
warnings.filterwarnings('ignore')


import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize(dataset, col_name):
    for i in dataset.columns:
        if i!=col_name:
            dataset[i] = (dataset[i]-dataset[i].mean())/(dataset[i].std())
    return dataset


def call_manan_AR(dataset, data, x, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, order_user, order_user_num):
    return mananbase.main_automate(dataset, data, x, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, order_user, order_user_num)

def call_manan_with_causality_AR(dataset, data, x, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, order_user, order_user_num):
    return manancausal.main_automate(dataset, data, x, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, order_user, order_user_num)

def call_mymethod1_AR(dataset, data, x, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, alpha_weights, alpha_sum, alpha_multiplier, type_of_model, order_user, order_user_num):
    return mymethod1.main_automate(dataset,  data, x, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, alpha_weights, alpha_sum, alpha_multiplier, type_of_model, order_user, order_user_num)

def call_mymethod3_AR(dataset, data, x, denorm_dict, T, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, fixed_ft, user_choice_subset):
    return mymethod3.main_automate(dataset, data, x, denorm_dict, T, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, fixed_ft, user_choice_subset)

def call_mymethod5_AR(dataset, data, x, denorm_dict, T, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, fixed_ft, user_choice_subset, normalization, max_number_of_permutations):
    return mymethod5.main_automate(dataset = dataset,
        data = data,
        x = x,
        denorm_dict = denorm_dict,                        
        T = T,
        index_x = index_x,
        lr = learning_rate,
        _lambda = _lambda,
        max_iter = max_iterations,
        gap_between_prints = gap_between_prints,
        size_of_dataset = size_of_dataset,
        seed = seed,
        type_of_model= type_of_model,
        fixed_ft = fixed_ft,
        user_choice_subset = user_choice_subset,
        normalization=normalization,
        max_number_of_permutations = max_number_of_permutations)

def call_mymethod8_AR(dataset, data, x, denorm_dict, T, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, fixed_ft, user_choice_subset, normalization, max_number_of_permutations, user_no):
    return mymethod8.main_automate(dataset = dataset,
        data = data,
        x = x,
        denorm_dict = denorm_dict,
        T = T,
        index_x = index_x,
        lr = learning_rate,
        _lambda = _lambda,
        max_iter = max_iterations,
        gap_between_prints = gap_between_prints,
        size_of_dataset = size_of_dataset,
        seed = seed,
        type_of_model= type_of_model,
        fixed_ft = fixed_ft,
        user_choice_subset = user_choice_subset,
        normalization=normalization,
        max_number_of_permutations = max_number_of_permutations)

def call_mymethod10_AR(dataset, data, x, denorm_dict, T, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, fixed_ft, user_choice_subset, normalization, max_number_of_permutations, user_no):
    return mymethod10.main_automate(dataset = dataset,
        data = data,
        x = x,
        denorm_dict = denorm_dict,
        T = T,
        index_x = index_x,
        lr = learning_rate,
        _lambda = _lambda,
        max_iter = max_iterations,
        gap_between_prints = gap_between_prints,
        size_of_dataset = size_of_dataset,
        seed = seed,
        type_of_model= type_of_model,
        fixed_ft = fixed_ft,
        user_choice_subset = user_choice_subset,
        normalization=normalization,
        max_number_of_permutations = max_number_of_permutations,
        user_no=user_no)



# def get_preference_order(data, N, col_name):
#     order_user = ['']*N #used to store the user preference order from most preferred at lower index to least preferred in highest index in feature names
#     order_user_num = ['']*N #used to store the user preference order from most preferred at lower index to least preferred in highest index in index values
#     i=0
#     print("Ordering indices start from 1. For example, feature with order index 1 means it is highest priority, 2 is lesser priority,  and so on.")
#     for col in data.columns:
#         if(col!=col_name):
#             print("Feature "+str(i+1)+": "+col)
#             i=i+1
#     print("Enter in order the preference below:")
#     for i in range(1, N):
#         inp =int(input("Enter feature of preference " + str(i)+" (only the feature number): "))
#         order_user[i-1] = data.columns[inp-1] 
#         order_user_num[i-1] = inp #used to store the user preference order in the original order of the dataframe

#     return order_user, order_user_num

def get_col(data, N, col_name):
    order_user = [] #used to store the user preference order from most preferred at lower index to least preferred in highest index in feature names
    order_user_num = [] #used to store the user preference order from most preferred at lower index to least preferred in highest index in index values
    i=0
    for i, col in enumerate(data.columns):
        if(col!=col_name):
            order_user.append(col)
            order_user_num.append(i+1)

    return order_user, order_user_num



def run(ch_number_of_users=None,
        ch_dataset=None,
        type_of_model=None,
        learning_rate = 1e-3,
        _lambda = 0.7,
        max_iterations = 100,
        gap_between_prints = 100,
        size_of_dataset = 1000,
        seed = 3,
        alpha_weights = None,
        alpha_sum=1000,
        alpha_multiplier=100, #Should be >1     
        normalization = "zscore",
        max_number_of_permutations = 3):
    
    if ch_dataset is None:
        ch_dataset = int(input("""\n
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
    if(ch_dataset==1):
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/dataset1.csv")
    elif(ch_dataset==2):
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/dataset2.csv")
    elif(ch_dataset==3): #default is taken as dataset 3
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/dataset3.csv")
    elif(ch_dataset==4):
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/dataset4.csv")
    elif(ch_dataset==5): #default is taken as dataset 3
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/dataset5.csv")
    elif(ch_dataset==6):
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/dataset6.csv")
    elif(ch_dataset==7): #default is taken as dataset 3
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/dataset7.csv")
    elif(ch_dataset==8): #default is taken as dataset 3
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/adult.csv")
    elif(ch_dataset==9): #default is taken as dataset 3
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/compas.csv")
    elif(ch_dataset==10): #default is taken as dataset 3
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/give_me_some_credit.csv")
    elif(ch_dataset==11): #default is taken as dataset 3
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/heloc.csv")
    # index_x = int(input("Choose the index of datapoint you want to change (-1 if you want random) :"))
    # learning_rate=float(input("Choose the value of learning rate (0.01 if you want basic): "))
    # _lambda = float(input("Choose the value of lambda (0.1 if you want basic): "))
    # max_iterations = int(input("Choose the number of iterations to run for (100 if you want basic): "))
    
    
    

    if ch_number_of_users is None:
        ch_number_of_users = int(input("""\nHow many users do you want to run it for?: """))

    data=pd.read_csv(dataset).sample(size_of_dataset)
    data = data.reset_index(drop=True)
    dataset_name = os.path.splitext(os.path.basename(dataset))[0] #extract the name of the dataset used from the path given in dataset (no extension)
    N=len(data.columns)

    
    #output column name
    if(dataset_name[:-1]=="dataset"):
        col_name = 'Y'
    else:
        if(dataset_name=="adult"):
            col_name = 'income'
        elif(dataset_name=="compas"):
            col_name = 'score'
        elif(dataset_name=="give_me_some_credit"):
            col_name = 'SeriousDlqin2yrs'
        elif(dataset_name=="heloc"):
            col_name = 'RiskPerformance'


    denorm_dict={}
    if normalization=="minmax":
        #for min-max keep track of min [0] and max [1]
        for var in data.columns:
            denorm_dict[var]=(data[var].min(), data[var].max())
    elif normalization=="zscore":
        #for z-score keep track of Mean [0] and SD [1]
        for var in data.columns:
            denorm_dict[var]=(data[var].mean(), data[var].std())

    

    number_of_zeroes = len(data.index[data[col_name]==0].tolist())
    if(ch_number_of_users>number_of_zeroes):
        ch_number_of_users = number_of_zeroes

    print("Number of users who have been classified as zero = ", number_of_zeroes)

    data = mymethod8.normalize(data, col_name, normalization)
    _, col_user_num = get_col(data, N, col_name)

    if type_of_model is None:
        type_of_model = int(input("""Choose the type of Model:
                        \n1) Custom Sequential 1
                        \n2) Custom Sequential 2
                        \n3) Custom Sequential 3
                        \n4) MLP Classifier
                        \nChoose the number: """))



    all_ft = list(data.drop([col_name], axis=1).columns)
    logger.debug('{}: all_ft = {}.'.format(datetime.datetime.now(), all_ft))
    subsets_for_fixed_ft_M3 = mymethod3.sublists_of_size_N(all_ft, N=1) + mymethod3.sublists_of_size_N(all_ft, N=2) #at max two features fixed
    logger.debug('{}: subsets_for_fixed_ft_M3 = {}.'.format(datetime.datetime.now(), subsets_for_fixed_ft_M3))
    # all_subsets = mymethod3.sublists(list(data.drop([col_name], axis=1).columns))
    # eligible_user_choice_subsets = list(set(tuple(row) for row in all_subsets) - set(tuple(row) for row in subsets_for_fixed_ft_M3))
    


    if(type_of_model==1):
        if(dataset_name=='dataset1'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential1_model_D1.pt")
        elif(dataset_name=='dataset2'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential1_model_D2.pt")
        elif(dataset_name=='dataset3'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential1_model_D3.pt")
        elif(dataset_name=='dataset4'):
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
    elif(type_of_model==2):
        if(dataset_name=='dataset1'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential2_model_D1.pt")
        elif(dataset_name=='dataset2'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential2_model_D2.pt")
        elif(dataset_name=='dataset3'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential2_model_D3.pt")
        elif(dataset_name=='dataset4'):
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
    elif(type_of_model==3):
        if(dataset_name=='dataset1'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential3_model_D1.pt")
        elif(dataset_name=='dataset2'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential3_model_D2.pt")
        elif(dataset_name=='dataset3'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/custom_sequential3_model_D3.pt")
        elif(dataset_name=='dataset4'):
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
    elif(type_of_model==4): #default is taken as MLP model
        if(dataset_name=='dataset1'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/mlp_classifier_model_D1.pt")
        elif(dataset_name=='dataset2'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/mlp_classifier_model_D2.pt")
        elif(dataset_name=='dataset3'):
            filename=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models/mlp_classifier_model_D3.pt")
        elif(dataset_name=='dataset4'):
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


    number_of_methods = 7
    violation_list = [0]*number_of_methods
    changed_ft = [0]*number_of_methods
    cost_cf = [0]*number_of_methods
    validity_list = [ch_number_of_users]*number_of_methods
    count = [0]*number_of_methods
    number_of_comp = [0]*number_of_methods
    ups = [0]*number_of_methods
    time_taken = [0.0]*number_of_methods

    method10_graph_list = []

    for i in range(ch_number_of_users):

        # index_x = random.choice(data.index[data[col_name]==0].tolist())
        # while(data.iloc[index_x][col_name]!=0.0):
        #     index_x = random.choice(data.index[data[col_name]==0].tolist())

        index_x = random.choice(data.index[data[col_name]==0].tolist())
        datapoint = data.drop([col_name], axis=1).iloc[index_x].values.astype(np.float32)
        y = model(torch.tensor(datapoint))
        logger.info('{}: y_pred = {} at index {} with datapoint {}.'.format(datetime.datetime.now(), y.data, index_x, datapoint))
        while(y.data==1):
            logger.info('{}: y_pred= 0. So changing index.'.format(datetime.datetime.now()))
            index_x = random.choice(data.index[data[col_name]==0].tolist())
            y = model(torch.tensor(data.drop([col_name], axis=1).iloc[index_x].values.astype(np.float32)))
            logger.info('{}: y_pred = {} at index {} with datapoint.'.format(datetime.datetime.now(), y.data, index_x))
        logger.info('\n{}: Final y_pred = {} at index {}.'.format(datetime.datetime.now(), y.data, index_x))
        x = data.drop([col_name], axis=1).iloc[index_x].values.astype(np.float32)

        #Get user order
        order_user_num = col_user_num.copy()
        random.shuffle(order_user_num)
        order_user = []
        for t in order_user_num:
            order_user.append(data.columns[t-1])

        all_subsets = mymethod8.sublists(all_ft)
        user_choice_subset = random.choice(all_subsets)
        T = random.randint(10,100)
        fixed_ft = []

        logger.info("\n\n-----------------------------------------------------User number {}-----------------------------------------------------\n".format(i+1))
        # x = np.reshape(data.drop([col_name], axis=1).iloc[index_x].tolist(), (1,N-1))[0]

        # logger.info('{}: index_x in compare = {}.'.format(datetime.datetime.now(), index_x))
        # logger.info('{}: x = {}.'.format(datetime.datetime.now(), x))
        # logger.info('{}: y = {}.'.format(datetime.datetime.now(), data.iloc[index_x][col_name]))
        
        method_num = 0
        logger.info("-----------------------------------------------------Manan's Base paper-----------------------------------------------------\n".format(i+1))
        violations, number_of_features_changed, cost_of_counterfactual, times = call_manan_AR(dataset, data, x, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, order_user, order_user_num)
        if number_of_features_changed!=-1:
            # print(violations)
            logger.debug('{}: Preference Order {}'.format(datetime.datetime.now(), order_user))
            ups[method_num] += 1 - (violations)/(math.factorial(N-1))
            violation_list[method_num] += violations
            changed_ft[method_num] += number_of_features_changed
            cost_cf[method_num] += cost_of_counterfactual
            count[method_num] += 1
        else:
            # logger.info('NO COUNTERFACTUAL FOUND FOR THIS METHOD BY MANAN"S BASE METHOD.')
            validity_list[method_num] -= 1
        time_taken[method_num] += times

        method_num=1
        logger.info("\n-----------------------------------------------------Manan's Base paper with Causality-----------------------------------------------------\n".format(i+1))
        violations, number_of_features_changed, cost_of_counterfactual, times = call_manan_with_causality_AR(dataset, data, x, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, order_user, order_user_num)
        if number_of_features_changed!=-1:
            # print(violations)
            ups[method_num] += 1 - (violations)/(math.factorial(N-1))
            violation_list[method_num] += violations
            changed_ft[method_num] += number_of_features_changed
            cost_cf[method_num] += cost_of_counterfactual
            count[method_num] += 1
        else:
            # logger.info('NO COUNTERFACTUAL FOUND FOR THIS METHOD BY MANAN"S METHOD WITH COUNTERFACTUAL.')
            validity_list[method_num] -= 1
        time_taken[method_num] += times

        method_num=2
        logger.info("\n-----------------------------------------------------My First Method-----------------------------------------------------\n".format(i+1))
        violations, number_of_features_changed, cost_of_counterfactual, times = call_mymethod1_AR(dataset, data, x, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, alpha_weights, alpha_sum, alpha_multiplier, type_of_model, order_user, order_user_num)
        if number_of_features_changed!=-1:
            # print(violations)
            ups[method_num] += 1 - (violations)/(math.factorial(N-1))
            violation_list[method_num] += violations
            changed_ft[method_num] += number_of_features_changed
            cost_cf[method_num] += cost_of_counterfactual
            count[method_num] += 1
        else:
            # logger.info('NO COUNTERFACTUAL FOUND FOR THIS METHOD BY MY FIRST METHOD.')
            validity_list[method_num] -= 1
        time_taken[method_num] += times

        method_num = 3
        logger.info("\n-----------------------------------------------------My Third Method-----------------------------------------------------\n".format(i+1))
        
        # fixed_ft = random.choice(subsets_for_fixed_ft_M3)
        # all_ft_copy = all_ft.copy()
        # for j in fixed_ft:
        #     all_ft_copy.remove(j)
        # remaining_subsets = mymethod3.sublists(all_ft_copy)
        # user_choice_subset = random.choice(remaining_subsets)
        number_of_features_changed, cost_of_counterfactual, times, comp, chosen_subset = call_mymethod3_AR(dataset, data, x, denorm_dict, T, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, fixed_ft, user_choice_subset)
        # ups[3] = 0 => NA for this method
        if number_of_features_changed!=-1:
            # print(violations)
            logger.debug('{}: Fixed features {}, Chosen subset {}, Output subset {}, Number of comparisons {}'.format(datetime.datetime.now(), fixed_ft, user_choice_subset, chosen_subset, comp))
            changed_ft[method_num] += number_of_features_changed
            cost_cf[method_num] += cost_of_counterfactual
            count[method_num] += 1
            number_of_comp[method_num] += comp
            if len(set(user_choice_subset) - set(chosen_subset))>0: #There are elements in actual user choice that have not been picked up
                violation_list[method_num] += 1
            if set(fixed_ft) & (set(chosen_subset)): #The fixed features should not come up in chosen subset - if true, it means there is intersection
                violation_list[method_num] += 1
        else:
            # logger.info('NO COUNTERFACTUAL FOUND FOR THIS METHOD BY MY FIRST METHOD.')
            validity_list[method_num] -= 1
        time_taken[method_num] += times
        

        # method_num=4
        # logger.info("\n-----------------------------------------------------My Fifth Method-----------------------------------------------------\n".format(i+1))
        # # fixed_ft = random.choice(subsets_for_fixed_ft_M3)
        # # all_ft_copy = all_ft.copy()
        # # for j in fixed_ft:
        # #     all_ft_copy.remove(j)
        # # remaining_subsets = mymethod3.sublists(all_ft_copy)
        # # user_choice_subset = random.choice(remaining_subsets)
        # # T = random.randint(10,100)
        # number_of_features_changed, cost_of_counterfactual, times, comp, chosen_subset = call_mymethod5_AR(dataset, data, x, denorm_dict, T, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, fixed_ft, user_choice_subset, normalization, max_number_of_permutations)
        # # ups[3] = 0 => NA for this method
        # if number_of_features_changed!=-1:
        #     # print(violations)
        #     logger.debug('{}: Fixed features {}, Chosen subset {}, Output subset {}, Number of comparisons {}'.format(datetime.datetime.now(), fixed_ft, user_choice_subset, chosen_subset, comp))
        #     changed_ft[method_num] += number_of_features_changed
        #     cost_cf[method_num] += cost_of_counterfactual
        #     count[method_num] += 1
        #     number_of_comp[method_num] += comp
        #     if len(set(user_choice_subset) - set(chosen_subset))>0: #There are elements in actual user choice that have not been picked up
        #         violation_list[method_num] += 1
        #     if set(fixed_ft) & (set(chosen_subset)): #The fixed features should not come up in chosen subset - if true, it means there is intersection
        #         violation_list[method_num] += 1
        # else:
        #     # logger.info('NO COUNTERFACTUAL FOUND FOR THIS METHOD BY MY FIRST METHOD.')
        #     validity_list[method_num] -= 1
        # time_taken[method_num] += times

        
        # method_num=5
        # logger.info("\n-----------------------------------------------------My Eight Method-----------------------------------------------------\n".format(i+1))
        # # fixed_ft = random.choice(subsets_for_fixed_ft_M3)
        # # all_ft_copy = all_ft.copy()
        # # for j in fixed_ft:
        # #     all_ft_copy.remove(j)
        # # remaining_subsets = mymethod3.sublists(all_ft_copy)
        # # user_choice_subset = random.choice(remaining_subsets)
        # # T = random.randint(10,100)
        # number_of_features_changed, cost_of_counterfactual, times, comp, chosen_subset = call_mymethod8_AR(dataset, data, x, denorm_dict, T, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, fixed_ft, user_choice_subset, normalization, max_number_of_permutations)
        # # ups[3] = 0 => NA for this method
        # if number_of_features_changed!=-1:
        #     # print(violations)
        #     logger.debug('{}: Fixed features {}, Chosen subset {}, Output subset {}, Number of comparisons {}'.format(datetime.datetime.now(), fixed_ft, user_choice_subset, chosen_subset, comp))
        #     changed_ft[method_num] += number_of_features_changed
        #     cost_cf[method_num] += cost_of_counterfactual
        #     count[method_num] += 1
        #     number_of_comp[method_num] += comp
        #     if len(set(user_choice_subset) - set(chosen_subset))>0: #There are elements in actual user choice that have not been picked up
        #         violation_list[method_num] += 1
        #     if set(fixed_ft) & (set(chosen_subset)): #The fixed features should not come up in chosen subset - if true, it means there is intersection
        #         violation_list[method_num] += 1
        # else:
        #     # logger.info('NO COUNTERFACTUAL FOUND FOR THIS METHOD BY MY FIRST METHOD.')
        #     validity_list[method_num] -= 1
        # time_taken[method_num] += times



        method_num=6
        logger.info("\n-----------------------------------------------------My Tenth Method-----------------------------------------------------\n".format(i+1))
        # fixed_ft = random.choice(subsets_for_fixed_ft_M3)
        # all_ft_copy = all_ft.copy()
        # for j in fixed_ft:
        #     all_ft_copy.remove(j)
        # remaining_subsets = mymethod3.sublists(all_ft_copy)
        # user_choice_subset = random.choice(remaining_subsets)
        # T = random.randint(10,100)
        number_of_features_changed, cost_of_counterfactual, times, comp, chosen_subset, tup_graph = call_mymethod10_AR(dataset, data, x, denorm_dict, T, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, fixed_ft, user_choice_subset, normalization, max_number_of_permutations, i+1)
        # ups[3] = 0 => NA for this method
        if number_of_features_changed!=-1:
            algo_chosen_subset=[]
            for t in chosen_subset:
                algo_chosen_subset.append(data.columns[t-1])
            logger.debug('{}: Fixed features {}, Chosen subset {}, Output subset {}, Number of comparisons {}'.format(datetime.datetime.now(), fixed_ft, user_choice_subset, chosen_subset, comp))
            changed_ft[method_num] += number_of_features_changed
            cost_cf[method_num] += cost_of_counterfactual
            count[method_num] += 1
            number_of_comp[method_num] += comp
            if len(set(user_choice_subset) - set(algo_chosen_subset))>0: #There are elements in actual user choice that have not been picked up
                violation_list[method_num] += 1
            if set(fixed_ft) & (set(algo_chosen_subset)): #The fixed features should not come up in chosen subset - if true, it means there is intersection
                violation_list[method_num] += 1
            method10_graph_list.append(tup_graph)
        else:
            # logger.info('NO COUNTERFACTUAL FOUND FOR THIS METHOD BY MY FIRST METHOD.')
            validity_list[method_num] -= 1
        time_taken[method_num] += times
        
        logger.info("\n---------------------------------------------------------------------------------------------------------------\n")
    
        



        # #print
        # logger.info('############################ITERATION INFO############################\n')
        # logger.info('{}: User datapoint is {}.\n'.format(datetime.datetime.now(), x))
        # for t in range(0,number_of_methods):
        #     if t == 0:
        #         logger.info('{}: Manan\'s Base Paper:'.format(datetime.datetime.now()))
        #     elif t == 1:
        #         logger.info('{}: Manan\'s Paper with Causality:'.format(datetime.datetime.now()))
        #     elif t == 2:
        #         logger.info('{}: My First Method:'.format(datetime.datetime.now()))
        #     elif t == 3:
        #         logger.info('{}: My Third Method:'.format(datetime.datetime.now()))
        #     elif t == 4:
        #         logger.info('{}: My Fifth Method:'.format(datetime.datetime.now()))
        #     elif t == 5:
        #         logger.info('{}: My Eight Method:'.format(datetime.datetime.now()))
        #     elif t == 6:
        #         logger.info('{}: My Tenth Method:'.format(datetime.datetime.now()))
        #     logger.info('{}: Total number of cases with  violations is {}.'.format(datetime.datetime.now(), violation_list[t]))
        #     logger.info('{}: Total UPS is {}.'.format(datetime.datetime.now(), ups[t]))
        #     logger.info('{}: Total number of features changed is {}.'.format(datetime.datetime.now(), changed_ft[t]))
        #     logger.info('{}: Total cost of counterfactuals is {}.'.format(datetime.datetime.now(), cost_cf[t]))
        #     logger.info('{}: Total comparisons is {}.'.format(datetime.datetime.now(), number_of_comp[t]))
    
    #print
    logger.info('############################FINAL INFO############################\n')
    logger.info('{}: Number of users run on is {}.\n'.format(datetime.datetime.now(), ch_number_of_users))
    for i in range(0,number_of_methods):
        # logger.debug('{}: Total number of violations is {}.'.format(datetime.datetime.now(), violation_list[i]))
        # logger.debug('{}: Total UPS is {}.'.format(datetime.datetime.now(), ups[i]))
        # logger.debug('{}: Total number of features changed is {}.'.format(datetime.datetime.now(), changed_ft[i]))
        # logger.debug('{}: Total cost of counterfactuals is {}.'.format(datetime.datetime.now(), cost_cf[i]))
        # logger.debug('{}: Total comparisons is {}.'.format(datetime.datetime.now(), number_of_comp[i]))

        if count[i]!=0:
            ups[i] = ups[i]/count[i]
            violation_list[i] = violation_list[i]/count[i]
            changed_ft[i] = changed_ft[i]/count[i]
            cost_cf[i] = cost_cf[i]/count[i]
            number_of_comp[i] = number_of_comp[i]/count[i]
        validity_list[i] = validity_list[i]/ch_number_of_users
        time_taken[i] = time_taken[i]/ch_number_of_users
        
        if i == 0:
            logger.info('{}: Manan\'s Base Paper:'.format(datetime.datetime.now()))
            logger.info('{}: Average number of cases with violations is {}.'.format(datetime.datetime.now(), violation_list[i]))
            logger.info('{}: Average UPS is {}.'.format(datetime.datetime.now(), ups[i]))
        elif i == 1:
            logger.info('{}: Manan\'s Paper with Causality:'.format(datetime.datetime.now()))
            logger.info('{}: Average number of cases with violations is {}.'.format(datetime.datetime.now(), violation_list[i]))
            logger.info('{}: Average UPS is {}.'.format(datetime.datetime.now(), ups[i]))
        elif i == 2:
            logger.info('{}: My First Method:'.format(datetime.datetime.now()))
            logger.info('{}: Average number of cases with violations is {}.'.format(datetime.datetime.now(), violation_list[i]))
            logger.info('{}: Average UPS is {}.'.format(datetime.datetime.now(), ups[i]))
        elif i == 3:
            logger.info('{}: My Third Method:'.format(datetime.datetime.now()))
            logger.info('{}: Average number of cases it failed to detect user preference/changed fixed features is {}.'.format(datetime.datetime.now(), violation_list[i]))
            logger.info('{}: Average number of comparisons is {}.'.format(datetime.datetime.now(), number_of_comp[i]))
        elif i == 4:
            logger.info('{}: My Fifth Method:'.format(datetime.datetime.now()))
            logger.info('{}: Average number of cases it failed to detect user preference/changed fixed features is {}.'.format(datetime.datetime.now(), violation_list[i]))
            logger.info('{}: Average number of comparisons is {}.'.format(datetime.datetime.now(), number_of_comp[i]))
        elif i == 5:
            logger.info('{}: My Eight Method:'.format(datetime.datetime.now()))
            logger.info('{}: Average number of cases it failed to detect user preference/changed fixed features is {}.'.format(datetime.datetime.now(), violation_list[i]))
            logger.info('{}: Average number of comparisons is {}.'.format(datetime.datetime.now(), number_of_comp[i]))
        elif i == 6:
            logger.info('{}: My Tenth Method:'.format(datetime.datetime.now()))
            logger.info('{}: Average number of cases it failed to detect user preference/changed fixed features is {}.'.format(datetime.datetime.now(), violation_list[i]))
            logger.info('{}: Average number of comparisons is {}.'.format(datetime.datetime.now(), number_of_comp[i]))
        
        logger.info('{}: Average number of features changed is {}.'.format(datetime.datetime.now(), changed_ft[i]))
        logger.info('{}: Average cost of counterfactuals is {}.'.format(datetime.datetime.now(), cost_cf[i]))
        logger.info('{}: Validity percentage is {}.'.format(datetime.datetime.now(), validity_list[i]*100))
        logger.info('{}: Average time taken to execute is {}.\n'.format(datetime.datetime.now(), time_taken[i]))

    #Graphs for Method10 for all users
    folder_name = datetime.datetime.now().strftime(f"{exp_name}_%B%d%Y_%I%M%p")
    folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'../../results/graphs/{folder_name}')
    if not os.path.exists(folder_path): #create folder if it doesnt exist
        os.makedirs(folder_path)
    
    method10_mincost_list = []
    method10_y_hat_list = []
    method10_list_T_list = []

    for t in method10_graph_list:
        list_mincost_b_hat, list_yhat_b_hat, list_T = t
        method10_mincost_list.append(list_mincost_b_hat)
        method10_y_hat_list.append(list_yhat_b_hat)
        method10_list_T_list.append(list_T)
    
    filename1=os.path.join(os.path.dirname(os.path.realpath(__file__)), f'../../results/graphs/{folder_name}/cost_bhat_vs_comparisons_AllUser_{exp_name}.png')
    if os.path.isfile(filename1):
        os.remove(filename1)
    filename2=os.path.join(os.path.dirname(os.path.realpath(__file__)), f'../../results/graphs/{folder_name}/y_hat_vs_comparisons_AllUser_{exp_name}.png')
    if os.path.isfile(filename2):
        os.remove(filename2)
    
    for i,_ in enumerate(method10_mincost_list):
        plt.plot(method10_list_T_list[i], method10_mincost_list[i], label=f'Cost of bhat v/s comparisons')
        plt.ylabel('Cost')
        plt.xlabel('Comparisons')
    plt.savefig(filename1)
    plt.clf() #clear the figure for the next graph
    for i,_ in enumerate(method10_y_hat_list):
        plt.plot(method10_list_T_list[i], method10_y_hat_list[i], label=f'Y_hat v/s comparisons')
        plt.ylabel('Y_hat')
        plt.xlabel('Comparisons')
    plt.savefig(filename2)
    plt.close()





if __name__=="__main__":
    time_start = time()
    ch_number_of_users = 100
    ch_dataset = 4
    type_of_model = 3

    learning_rate = 1e-3
    _lambda = 0.7
    max_iterations = 100
    gap_between_prints = 100
    size_of_dataset = 1000
    seed = 3
    alpha_weights = None
    alpha_sum=1000
    alpha_multiplier=100 #Should be >1
    normalization = "zscore"
    max_number_of_permutations = 3

    run(ch_number_of_users=ch_number_of_users,
        ch_dataset=ch_dataset,
        type_of_model=type_of_model,
        learning_rate = learning_rate,
        _lambda = _lambda,
        max_iterations = max_iterations,
        gap_between_prints = gap_between_prints,
        size_of_dataset = size_of_dataset,
        seed = seed,
        alpha_weights = alpha_weights,
        alpha_sum=alpha_sum,
        alpha_multiplier=alpha_multiplier,  
        normalization = normalization,
        max_number_of_permutations = max_number_of_permutations)


    time_end = time()
    time_taken = time_end-time_start
    logger.info('\n**********************************************************************\n{}: Total time taken to execute this script is {}.\n'.format(datetime.datetime.now(), time_taken))
