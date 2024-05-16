import os
import sys
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
import pandas as pd
import numpy as np
from counterfactual_gen import  Algorithm10 as mymethod10
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../"))
import datetime
from time import time
import random



import logging
exp_name = "give_me_credit_alg10"
log_file_name = "Log_" + exp_name + ".log"
log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../logs", log_file_name)
logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(log_path, mode='w')
# fh.setLevel(logging.DEBUG)
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

import warnings
warnings.filterwarnings('ignore')


import torch
import torch.nn as nn
import torch.nn.functional as F



def call_mymethod10_AR(dataset, data, x, denorm_dict, T, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, fixed_ft, user_choice_subset, normalization, max_number_of_permutations, user_no=1, print_graph=0):
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
        user_no=user_no,
        print_graph=print_graph)


def run(ch_dataset=None, ch_number_of_users=None, type_of_model=None):
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
    
    
    learning_rate = 1e-3
    _lambda = 0.7
    max_iterations = 100
    gap_between_prints = 99
    size_of_dataset = 200
    seed = 3
    normalization = "zscore"
    max_number_of_permutations = 3

    if ch_number_of_users is None:
        ch_number_of_users = int(input("""\nHow many users do you want to run it for?: """))

    data=pd.read_csv(dataset).head(size_of_dataset)
    data = data.reset_index(drop=True) #Resetting index for the current dataframe otherwise random indices come up which leads to out of bounds error. Drop = True ensures older index are not added as new column
    print("Size of dataset = ", len(data))
    dataset_name = os.path.splitext(os.path.basename(dataset))[0] #extract the name of the dataset used from the path given in dataset (no extension)
    N=len(data.columns)

    #Keep track of original values so that we can denormalize
    denorm_dict={}
    if normalization=="minmax":
        #for min-max keep track of min [0] and max [1]
        for var in data.columns:
            denorm_dict[var]=(data[var].min(), data[var].max())
    elif normalization=="zscore":
        #for z-score keep track of Mean [0] and SD [1]
        for var in data.columns:
            denorm_dict[var]=(data[var].mean(), data[var].std())
    
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

    #as of now assume data is not normalized by traditional techniques
    data = mymethod10.normalize(data, col_name, normalization)

    number_of_zeroes = len(data.index[data[col_name]==0].tolist())
    if(ch_number_of_users>number_of_zeroes):
        ch_number_of_users = number_of_zeroes

    print("Number of users who have been classified as zero = ", number_of_zeroes)

    if type_of_model is None:
        type_of_model = int(input("""Choose the type of Model:
                        \n1) Custom Sequential 1
                        \n2) Custom Sequential 2
                        \n3) Custom Sequential 3
                        \n4) MLP Classifier
                        \nChoose the number: """))

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
    


    violation_list = [0]*ch_number_of_users
    changed_ft = [0]*ch_number_of_users
    cost_cf = [0]*ch_number_of_users
    validity_list = [1]*ch_number_of_users
    number_of_comp = [0]*ch_number_of_users
    time_taken = [0.0]*ch_number_of_users

    all_ft = list(data.drop([col_name], axis=1).columns)
    logger.debug('{}: all_ft = {}.'.format(datetime.datetime.now(), all_ft))
    subsets_for_fixed_ft_M3 = mymethod10.sublists_till_size_N(all_ft, N=2) #at max five features fixed
    logger.debug('{}: subsets_for_fixed_ft_M3 = {}.'.format(datetime.datetime.now(), subsets_for_fixed_ft_M3))
    
    for i in range(ch_number_of_users):
        index_x = random.choice(data.index[data[col_name]==0].tolist())
        datapoint = data.drop([col_name], axis=1).iloc[index_x].values.astype(np.float32)
        y = model(torch.tensor(datapoint))
        logger.info('{}: y_pred = {} at index {} with datapoint {}.'.format(datetime.datetime.now(), y.data, index_x, datapoint))
        while(y.data==1):
            logger.info('{}: y_pred= 0. So changing index.'.format(datetime.datetime.now()))
            index_x = random.choice(data.index[data[col_name]==0].tolist())
            y = model(torch.tensor(data.drop([col_name], axis=1).iloc[index_x].values.astype(np.float32)))
            logger.info('{}: y_pred = {} at index {} with datapoint.'.format(datetime.datetime.now(), y.data, index_x))
        logger.info('{}: Final y_pred = {} at index {}.'.format(datetime.datetime.now(), y.data, index_x))
        x = data.drop([col_name], axis=1).iloc[index_x].values.astype(np.float32)
        

        logger.info("\n-----------------------------------------------------User {}-----------------------------------------------------\n".format(i+1))
        fixed_ft = random.choice(subsets_for_fixed_ft_M3)
        all_ft_copy = all_ft.copy()
        for j in fixed_ft:
            all_ft_copy.remove(j)
        remaining_subsets = mymethod10.sublists(all_ft_copy)
        user_choice_subset = random.choice(remaining_subsets)
        T = random.randint(10,100)
        logger.info('{}: User chosen subset = {}.'.format(datetime.datetime.now(), user_choice_subset))
        number_of_features_changed, cost_of_counterfactual, times, comp, chosen_subset, list_graph = call_mymethod10_AR(dataset, data, x, denorm_dict, T, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, fixed_ft, user_choice_subset, normalization, max_number_of_permutations, i+1, 0)
        
        if number_of_features_changed!=-1:
            algo_chosen_subset=[]
            for t in chosen_subset:
                algo_chosen_subset.append(data.columns[t-1])
            logger.info('{}: Fixed features {}, Chosen subset {}, Output subset {}, Number of comparisons {}'.format(datetime.datetime.now(), fixed_ft, user_choice_subset, algo_chosen_subset, comp))
            changed_ft[i] = number_of_features_changed
            cost_cf[i] = cost_of_counterfactual
            time_taken[i] = times
            number_of_comp[i] = comp
            logger.info('{}: set(user_choice_subset) - set(algo_chosen_subset) = {}, and its length ={}.'.format(datetime.datetime.now(), set(user_choice_subset) - set(algo_chosen_subset), len(set(user_choice_subset) - set(algo_chosen_subset))))
            if len(set(user_choice_subset) - set(algo_chosen_subset))>0: #There are elements in actual user choice that have not been picked up
                violation_list[i] += 1
            if set(fixed_ft) & (set(algo_chosen_subset)): #The fixed features should not come up in chosen subset - if true, it means there is intersection
                violation_list[i] += 1
        else:
            validity_list[i] = 0
        logger.info("\n---------------------------------------------------------------------------------------------------------------\n")
    
    logger.info('############################FINAL INFO############################\n')
    logger.info('{}: Number of users run on is {}.\n'.format(datetime.datetime.now(), ch_number_of_users))

    if sum(validity_list)==0:
        logger.info('{}: No valid counterfactuals generated.\n'.format(datetime.datetime.now()))
    else:
        violation_list_avg = sum(violation_list)/sum(validity_list) #only check for users who have a valid counterfactual
        changed_ft_avg = sum(changed_ft)/sum(validity_list) #only check for users who have a valid counterfactual
        cost_cf_avg = sum(cost_cf)/sum(validity_list) #only check for users who have a valid counterfactual
        validity_list_avg = sum(validity_list)/ch_number_of_users
        time_taken_avg = sum(time_taken)/ch_number_of_users
        number_of_comp_avg = sum(number_of_comp)/ch_number_of_users

        logger.info('{}: My Tenth Method:'.format(datetime.datetime.now()))
        logger.info('{}: Average number of cases it failed to detect user preference/changed fixed features is {}. ({})'.format(datetime.datetime.now(), violation_list_avg, violation_list))
        logger.info('{}: Average number of comparisons is {}. ({})'.format(datetime.datetime.now(), number_of_comp_avg, number_of_comp))

        logger.info('{}: Average number of features changed is {}. ({})'.format(datetime.datetime.now(), changed_ft_avg, changed_ft))
        logger.info('{}: Average cost of counterfactuals is {}. ({})'.format(datetime.datetime.now(), cost_cf_avg, cost_cf))
        logger.info('{}: Validity percentage is {}. ({})'.format(datetime.datetime.now(), validity_list_avg*100, validity_list))
        logger.info('{}: Average time taken to execute is {}. ({})\n'.format(datetime.datetime.now(), time_taken_avg, time_taken))

        folder_name = datetime.datetime.now().strftime(f"{exp_name}_multiple_%B%d%Y_%I%M%p")
        folder_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'../../results/graphs/{folder_name}')
        if not os.path.exists(folder_path): #create folder if it doesnt exist
            os.makedirs(folder_path)
        x_values = range(1, ch_number_of_users+1)
        plt.plot(x_values, changed_ft, label=f'Changed features for each User')
        plt.ylabel('Changes')
        plt.xlabel('User')
        filename1 = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'../../results/graphs/{folder_name}/plot_{exp_name}_changes_{ch_number_of_users}.png')
        if os.path.isfile(filename1):
            os.remove(filename1)
        plt.savefig(filename1)
        plt.plot(x_values, cost_cf, label=f'Cost for each User')
        plt.ylabel('Cost of Counterfactuals')
        plt.xlabel('User')
        filename2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'../../results/graphs/{folder_name}/plot_{exp_name}_cost_{ch_number_of_users}.png')
        if os.path.isfile(filename2):
            os.remove(filename2)
        plt.savefig(filename2)
        plt.close()



if __name__=="__main__":
    time_start = time()
    ch_dataset = 4
    ch_number_of_users = 5
    type_of_model = 3
    run(ch_dataset, ch_number_of_users, type_of_model)
    time_end = time()
    logger.info('{}: Time taken for script to run = {}:'.format(datetime.datetime.now(), time_end-time_start))

