import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
import pandas as pd
import numpy as np
from counterfactual_gen import  Algorithm5 as mymethod5
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../"))
import datetime
import random
import matplotlib.pyplot as plt


import logging
exp_name = "give_me_credit_alg5"
log_file_name = "Log_" + exp_name + ".log"
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



def call_mymethod5_AR(dataset, T, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, fixed_ft, user_choice_subset, normalization, max_number_of_permutations):
    return mymethod5.main(dataset = dataset,
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


def run():
    ch_number_of_users = int(input("""\n
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
    if(ch_number_of_users==1):
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/dataset1.csv")
    elif(ch_number_of_users==2):
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/dataset2.csv")
    elif(ch_number_of_users==3): #default is taken as dataset 3
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/dataset3.csv")
    elif(ch_number_of_users==4):
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/dataset4.csv")
    elif(ch_number_of_users==5): #default is taken as dataset 3
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/dataset5.csv")
    elif(ch_number_of_users==6):
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/dataset6.csv")
    elif(ch_number_of_users==7): #default is taken as dataset 3
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/dataset7.csv")
    elif(ch_number_of_users==8): #default is taken as dataset 3
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/adult.csv")
    elif(ch_number_of_users==9): #default is taken as dataset 3
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/compas.csv")
    elif(ch_number_of_users==10): #default is taken as dataset 3
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/give_me_some_credit.csv")
    elif(ch_number_of_users==11): #default is taken as dataset 3
        dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/heloc.csv")
    
    
    learning_rate = 1e-3
    _lambda = 0.7
    max_iterations = 100
    gap_between_prints = 100
    size_of_dataset = 5000
    seed = 3
    normalization = "zscore"
    max_number_of_permutations = 3

    ch_number_of_users = int(input("""\nHow many users do you want to run it for?: """))

    data=pd.read_csv(dataset).head(size_of_dataset)
    data = data.reset_index(drop=True) #Resetting index for the current dataframe otherwise random indices come up which leads to out of bounds error. Drop = True ensures older index are not added as new column
    print("size of dataset = ", len(data))
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

    logger.info(data[data[col_name]==0].to_markdown())

    number_of_zeroes = len(data.index[data[col_name]==0].tolist())
    if(ch_number_of_users>number_of_zeroes):
        ch_number_of_users = number_of_zeroes

    print("Number of users who have been classified as zero = ", number_of_zeroes)

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
    subsets_for_fixed_ft_M3 = mymethod5.sublists_till_size_N(all_ft, N=5) #at max five features fixed
    logger.debug('{}: subsets_for_fixed_ft_M3 = {}.'.format(datetime.datetime.now(), subsets_for_fixed_ft_M3))
    
    for i in range(ch_number_of_users):
        index_x = random.choice(data.index[data[col_name]==0].tolist())
        datapoint = data.drop([col_name], axis=1).iloc[index_x].values.astype(np.float32)
        y = model(torch.tensor(datapoint))
        while(y!=0.0):
            index_x = random.choice(data.index[data[col_name]==0].tolist())
            y = model(torch.tensor(data.drop([col_name], axis=1).iloc[index_x].values.astype(np.float32)))
            
        
        

        logger.info("\n-----------------------------------------------------User {}-----------------------------------------------------\n".format(i+1))
        fixed_ft = random.choice(subsets_for_fixed_ft_M3)
        all_ft_copy = all_ft.copy()
        for j in fixed_ft:
            all_ft_copy.remove(j)
        remaining_subsets = mymethod5.sublists(all_ft_copy)
        user_choice_subset = random.choice(remaining_subsets)
        T = random.randint(10,100)
        number_of_features_changed, cost_of_counterfactual, times, comp, chosen_subset = call_mymethod5_AR(dataset, T, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, fixed_ft, user_choice_subset, normalization, max_number_of_permutations)
        if number_of_features_changed!=-1:
            logger.info('{}: Fixed features {}, Chosen subset {}, Output subset {}, Number of comparisons {}'.format(datetime.datetime.now(), fixed_ft, user_choice_subset, chosen_subset, comp))
            changed_ft[i] = number_of_features_changed
            cost_cf[i] = cost_of_counterfactual
            time_taken[i] = times
            number_of_comp[i] = comp
            if set(chosen_subset) >= set(user_choice_subset): #The actual subset is a superset of the user choice subset
                violation_list[i] = 1 #Violation here implies user actual subset did not match
            elif set(fixed_ft) & (set(chosen_subset)): #The fixed features should not come up in chosen subset - if true, it means there is intersection
                violation_list[i] = 1
        else:
            validity_list[i] = 0
    
    logger.info('############################FINAL INFO############################\n')
    logger.info('{}: Number of users run on is {}.\n'.format(datetime.datetime.now(), ch_number_of_users))

    violation_list_avg = sum(violation_list)/ch_number_of_users
    changed_ft_avg = sum(changed_ft)/ch_number_of_users
    cost_cf_avg = sum(cost_cf)/ch_number_of_users
    validity_list_avg = sum(validity_list)/ch_number_of_users
    time_taken_avg = sum(time_taken)/ch_number_of_users
    number_of_comp_avg = sum(number_of_comp)/ch_number_of_users
    
    logger.info('{}: My Fifth Method:'.format(datetime.datetime.now()))
    logger.info('{}: Average number of times it failed to detect user preference/changed fixed features is {}.'.format(datetime.datetime.now(), violation_list_avg))
    logger.info('{}: Average number of comparisons is {}.'.format(datetime.datetime.now(), number_of_comp_avg))
    
    logger.info('{}: Average number of features changed is {}.'.format(datetime.datetime.now(), changed_ft_avg))
    logger.info('{}: Average cost of counterfactuals is {}.'.format(datetime.datetime.now(), cost_cf_avg))
    logger.info('{}: Validity percentage is {}.'.format(datetime.datetime.now(), validity_list_avg*100))
    logger.info('{}: Average time taken to execute is {}.\n'.format(datetime.datetime.now(), time_taken_avg))

    x_values = range(1, ch_number_of_users+1)
    plt.plot(x_values, changed_ft, label=f'Changed features for each User')
    plt.ylabel('Changes')
    plt.xlabel('User')
    plt.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)), f'./graphs/plot_{exp_name}_changes_50user_alg5.png'))
    plt.plot(x_values, cost_cf, label=f'Cost for each User')
    plt.ylabel('Cost of Counterfactuals')
    plt.xlabel('User')
    plt.savefig(os.path.join(os.path.dirname(os.path.realpath(__file__)), f'./graphs/plot_{exp_name}_cost_50user_alg5.png'))
    plt.close()



if __name__=="__main__":
    run()