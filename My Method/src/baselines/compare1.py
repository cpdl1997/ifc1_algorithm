import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
import pandas as pd
from counterfactual_gen import  manan_algorithmic_recourse as mananbase, manan_with_causality_algorithmic_recourse as manancausal, my_method_algorithmic_recourse as mymethod1, my_method_algorithmic_recourse3 as mymethod3, Algorithm5 as mymethod5
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../"))
import datetime
import random
import math


import logging
log_file_name = "Log_comparison1.log"
log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../logs", log_file_name)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)
fh = logging.FileHandler(log_path, mode='w')
fh.setLevel(logging.DEBUG)
# fh.setLevel(logging.INFO)
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


def call_manan_AR(dataset,index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, order_user, order_user_num):
    return mananbase.main(dataset, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, order_user, order_user_num)

def call_manan_with_causality_AR(dataset,index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, order_user, order_user_num):
    return manancausal.main(dataset, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, order_user, order_user_num)

def call_mymethod1_AR(dataset,index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, alpha_weights, alpha_sum, alpha_multiplier, type_of_model, order_user, order_user_num):
    return mymethod1.main(dataset, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, alpha_weights, alpha_sum, alpha_multiplier, type_of_model, order_user, order_user_num)

def call_mymethod3_AR(dataset, T, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, fixed_ft, user_choice_subset):
    return mymethod3.main(dataset, T, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, fixed_ft, user_choice_subset)

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
    # index_x = int(input("Choose the index of datapoint you want to change (-1 if you want random) :"))
    # learning_rate=float(input("Choose the value of learning rate (0.01 if you want basic): "))
    # _lambda = float(input("Choose the value of lambda (0.1 if you want basic): "))
    # max_iterations = int(input("Choose the number of iterations to run for (100 if you want basic): "))
    
    
    learning_rate = 1e-3
    _lambda = 0.7
    max_iterations = 100
    gap_between_prints = 100
    size_of_dataset = 5000
    seed = 3
    alpha_weights = None
    alpha_sum=1000
    alpha_multiplier=100 #Should be >1
    normalization = "zscore"
    max_number_of_permutations = 3

    ch_number_of_users = int(input("""\nHow many users do you want to run it for?: """))

    data=pd.read_csv(dataset).sample(size_of_dataset)
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


    number_of_zeroes = len(data.index[data[col_name]==0].tolist())
    if(ch_number_of_users>number_of_zeroes):
        ch_number_of_users = number_of_zeroes

    print("Number of users who have been classified as zero = ", number_of_zeroes)

    data = normalize(data, col_name)
    order_user, order_user_num = get_preference_order(data, N, col_name)


    type_of_model = int(input("""Choose the type of Model:
                    \n1) Custom Sequential 1
                    \n2) Custom Sequential 2
                    \n3) Custom Sequential 3
                    \n4) MLP Classifier
                    \nChoose the number: """))


    number_of_methods = 5
    violation_list = [0]*number_of_methods
    changed_ft = [0]*number_of_methods
    cost_cf = [0]*number_of_methods
    validity_list = [ch_number_of_users]*number_of_methods
    count = [0]*number_of_methods
    number_of_comp = [0]*number_of_methods
    ups = [0]*number_of_methods
    time_taken = [0.0]*number_of_methods

    all_ft = list(data.drop([col_name], axis=1).columns)
    logger.debug('{}: all_ft = {}.'.format(datetime.datetime.now(), all_ft))
    subsets_for_fixed_ft_M3 = mymethod3.sublists_of_size_N(all_ft, N=1) + mymethod3.sublists_of_size_N(all_ft, N=2) #at max two features fixed
    logger.debug('{}: subsets_for_fixed_ft_M3 = {}.'.format(datetime.datetime.now(), subsets_for_fixed_ft_M3))
    # all_subsets = mymethod3.sublists(list(data.drop([col_name], axis=1).columns))
    # eligible_user_choice_subsets = list(set(tuple(row) for row in all_subsets) - set(tuple(row) for row in subsets_for_fixed_ft_M3))
    


    for i in range(ch_number_of_users):

        index_x = random.choice(data.index[data[col_name]==0].tolist())
        while(data.iloc[index_x][col_name]!=0.0):
            index_x = random.choice(data.index[data[col_name]==0].tolist())

        

        logger.info("\n\n-----------------------------------------------------User number {}-----------------------------------------------------\n".format(i+1))
        # x = np.reshape(data.drop([col_name], axis=1).iloc[index_x].tolist(), (1,N-1))[0]

        # logger.info('{}: index_x in compare = {}.'.format(datetime.datetime.now(), index_x))
        # logger.info('{}: x = {}.'.format(datetime.datetime.now(), x))
        # logger.info('{}: y = {}.'.format(datetime.datetime.now(), data.iloc[index_x][col_name]))
        
        logger.info("-----------------------------------------------------Manan's Base paper-----------------------------------------------------\n".format(i+1))
        violations, number_of_features_changed, cost_of_counterfactual, times = call_manan_AR(dataset, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, order_user, order_user_num)
        if number_of_features_changed!=-1:
            # print(violations)
            ups[0] += 1 - (violations)/(math.factorial(N-1))
            violation_list[0] += violations
            changed_ft[0] += number_of_features_changed
            cost_cf[0] += cost_of_counterfactual
            count[0] += 1
            time_taken[0] += times
        else:
            # logger.info('NO COUNTERFACTUAL FOUND FOR THIS METHOD BY MANAN"S BASE METHOD.')
            validity_list[0] -= 1

        logger.info("\n-----------------------------------------------------Manan's Base paper with Causality-----------------------------------------------------\n".format(i+1))
        violations, number_of_features_changed, cost_of_counterfactual, times = call_manan_with_causality_AR(dataset, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, order_user, order_user_num)
        if number_of_features_changed!=-1:
            # print(violations)
            ups[1] += 1 - (violations)/(math.factorial(N-1))
            violation_list[1] += violations
            changed_ft[1] += number_of_features_changed
            cost_cf[1] += cost_of_counterfactual
            count[1] += 1
            time_taken[1] += times
        else:
            # logger.info('NO COUNTERFACTUAL FOUND FOR THIS METHOD BY MANAN"S METHOD WITH COUNTERFACTUAL.')
            validity_list[1] -= 1

        logger.info("\n-----------------------------------------------------My First Method-----------------------------------------------------\n".format(i+1))
        violations, number_of_features_changed, cost_of_counterfactual, times = call_mymethod1_AR(dataset, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, alpha_weights, alpha_sum, alpha_multiplier, type_of_model, order_user, order_user_num)
        if number_of_features_changed!=-1:
            # print(violations)
            ups[2] += 1 - (violations)/(math.factorial(N-1))
            violation_list[2] += violations
            changed_ft[2] += number_of_features_changed
            cost_cf[2] += cost_of_counterfactual
            count[2] += 1
            time_taken[2] += times
        else:
            # logger.info('NO COUNTERFACTUAL FOUND FOR THIS METHOD BY MY FIRST METHOD.')
            validity_list[2] -= 1

        logger.info("\n-----------------------------------------------------My Third Method-----------------------------------------------------\n".format(i+1))
        fixed_ft = random.choice(subsets_for_fixed_ft_M3)
        all_ft_copy = all_ft.copy()
        for j in fixed_ft:
            all_ft_copy.remove(j)
        remaining_subsets = mymethod3.sublists(all_ft_copy)
        user_choice_subset = random.choice(remaining_subsets)
        T = random.randint(10,100)
        number_of_features_changed, cost_of_counterfactual, times, comp, chosen_subset = call_mymethod3_AR(dataset, T, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, fixed_ft, user_choice_subset)
        # ups[3] = 0 => NA for this method
        if number_of_features_changed!=-1:
            # print(violations)
            logger.debug('{}: Fixed features {}, Chosen subset {}, Output subset {}, Number of comparisons {}'.format(datetime.datetime.now(), fixed_ft, user_choice_subset, chosen_subset, comp))
            changed_ft[3] += number_of_features_changed
            cost_cf[3] += cost_of_counterfactual
            count[3] += 1
            time_taken[3] += times
            number_of_comp[3] += comp
            if set(chosen_subset) >= set(user_choice_subset): #The actual subset is a superset of the user choice subset
                violation_list[3] += 1
            elif set(fixed_ft) & (set(chosen_subset)): #The fixed features should not come up in chosen subset - if true, it means there is intersection
                violation_list[3] += 1
        else:
            # logger.info('NO COUNTERFACTUAL FOUND FOR THIS METHOD BY MY FIRST METHOD.')
            validity_list[3] -= 1
        

        logger.info("\n-----------------------------------------------------My Fifth Method-----------------------------------------------------\n".format(i+1))
        fixed_ft = random.choice(subsets_for_fixed_ft_M3)
        all_ft_copy = all_ft.copy()
        for j in fixed_ft:
            all_ft_copy.remove(j)
        remaining_subsets = mymethod3.sublists(all_ft_copy)
        user_choice_subset = random.choice(remaining_subsets)
        T = random.randint(10,100)
        number_of_features_changed, cost_of_counterfactual, times, comp, chosen_subset = call_mymethod5_AR(dataset, T, index_x, learning_rate, _lambda, max_iterations, gap_between_prints, size_of_dataset, seed, type_of_model, fixed_ft, user_choice_subset, normalization, max_number_of_permutations)
        # ups[3] = 0 => NA for this method
        if number_of_features_changed!=-1:
            # print(violations)
            logger.debug('{}: Fixed features {}, Chosen subset {}, Output subset {}, Number of comparisons {}'.format(datetime.datetime.now(), fixed_ft, user_choice_subset, chosen_subset, comp))
            changed_ft[4] += number_of_features_changed
            cost_cf[4] += cost_of_counterfactual
            count[4] += 1
            time_taken[4] += times
            number_of_comp[4] += comp
            if set(chosen_subset) >= set(user_choice_subset): #The actual subset is a superset of the user choice subset
                violation_list[4] += 1
            elif set(fixed_ft) & (set(chosen_subset)): #The fixed features should not come up in chosen subset - if true, it means there is intersection
                violation_list[4] += 1
        else:
            # logger.info('NO COUNTERFACTUAL FOUND FOR THIS METHOD BY MY FIRST METHOD.')
            validity_list[4] -= 1
    
    #print
    logger.info('############################FINAL INFO############################\n')
    logger.info('{}: Number of users run on is {}.\n'.format(datetime.datetime.now(), ch_number_of_users))
    for i in range(0,number_of_methods):
        # logger.debug('{}: Total number of violations is {}.'.format(datetime.datetime.now(), violation_list[i]))
        # logger.debug('{}: Total UPS is {}.'.format(datetime.datetime.now(), ups[i]))
        # logger.debug('{}: Total number of features changed is {}.'.format(datetime.datetime.now(), changed_ft[i]))
        # logger.debug('{}: Total cost of counterfactuals is {}.'.format(datetime.datetime.now(), cost_cf[i]))
        # logger.debug('{}: Total comparisons is {}.'.format(datetime.datetime.now(), number_of_comp[i]))
        ups[i] = ups[i]/count[i]
        violation_list[i] = violation_list[i]/count[i]
        changed_ft[i] = changed_ft[i]/count[i]
        cost_cf[i] = cost_cf[i]/count[i]
        validity_list[i] = validity_list[i]/ch_number_of_users
        time_taken[i] = time_taken[i]/count[i]
        number_of_comp[i] = number_of_comp[i]/count[i]
        if i == 0:
            logger.info('{}: Manan\'s Base Paper:'.format(datetime.datetime.now()))
            logger.info('{}: Average number of violations is {}.'.format(datetime.datetime.now(), violation_list[i]))
            logger.info('{}: Average UPS is {}.'.format(datetime.datetime.now(), ups[i]))
        elif i == 1:
            logger.info('{}: Manan\'s Paper with Causality:'.format(datetime.datetime.now()))
            logger.info('{}: Average number of violations is {}.'.format(datetime.datetime.now(), violation_list[i]))
            logger.info('{}: Average UPS is {}.'.format(datetime.datetime.now(), ups[i]))
        elif i == 2:
            logger.info('{}: My First Method:'.format(datetime.datetime.now()))
            logger.info('{}: Average number of violations is {}.'.format(datetime.datetime.now(), violation_list[i]))
            logger.info('{}: Average UPS is {}.'.format(datetime.datetime.now(), ups[i]))
        elif i == 3:
            logger.info('{}: My Third Method:'.format(datetime.datetime.now()))
            logger.info('{}: Average number of times it failed to detect user preference/changed fixed features is {}.'.format(datetime.datetime.now(), violation_list[i]))
            logger.info('{}: Average number of comparisons is {}.'.format(datetime.datetime.now(), number_of_comp[i]))
        elif i == 4:
            logger.info('{}: My Fifth Method:'.format(datetime.datetime.now()))
            logger.info('{}: Average number of times it failed to detect user preference/changed fixed features is {}.'.format(datetime.datetime.now(), violation_list[i]))
            logger.info('{}: Average number of comparisons is {}.'.format(datetime.datetime.now(), number_of_comp[i]))
        
        logger.info('{}: Average number of features changed is {}.'.format(datetime.datetime.now(), changed_ft[i]))
        logger.info('{}: Average cost of counterfactuals is {}.'.format(datetime.datetime.now(), cost_cf[i]))
        logger.info('{}: Validity percentage is {}.'.format(datetime.datetime.now(), validity_list[i]*100))
        logger.info('{}: Average time taken to execute is {}.\n'.format(datetime.datetime.now(), time_taken[i]))






if __name__=="__main__":
    run()