"""
This is an implementation which uses duelling bandits to generate counterfactuals -> With the SCM module being automated using LiNGAM via 
the dowhy and causal-learn packages. This is a modification of algorithm 4 which considers the following:
1) Dataset used to train the model is not available to the outside world. So we call the model several times using randomized input features
to see how it behaves and collect the dataset for generating SCM
2) We consider categorical features using a one-hot encoding system
3) (if possible) Change the gradient learning system so that features not participating in the process dont have any gradients calculated
4) (later on) A change in the cost methodology
5) Updated metrics for comparison with baselines and settings for automated calling of this script

"""


import os
import glob
import sys
import math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
import pandas as pd
import numpy as np
# np.set_printoptions(precision=3) #only 3 decimal digits will be shown for any numpy array
from dataset_gen import dataset_generator as dg
# from model import custom_sequential, mlp_classifier
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")) #for causal-learn, dowhy and graphviz
from algorithms import SeventhMethod
from scm import SCM, Node
import util.QueueSet as QS
import util.PrioritySet as PS
import datetime
import random
import copy
import itertools
from time import time

import scm.graphviz as graphviz
from scm.dowhy import CausalModel
from scm.causallearn.search.FCMBased import lingam
from scm.causallearn.search.FCMBased.lingam.utils import make_dot


import logging
log_file_name = "Log_algorithm7.log"
log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../logs", log_file_name)
logger = logging.getLogger()
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(log_path, mode='w')
# fh.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)
logger.addHandler(fh)

import warnings
warnings.filterwarnings('ignore')


import torch
import torch.nn as nn
import torch.nn.functional as F


########################################### UTILITY FUNCTIONS ###########################################

#Given: List lst (list)
#Output: All sublists from size of 1 to N
def sublists(lst, index=0, current=[]):
    sub = []
    for L in range(len(lst)+1):
        for subset in itertools.combinations(lst, L):
            if subset:
                sub.append(list(subset))
        
    return sub

#Given: List lst (list)
#Output: All sublists of size N only
def sublists_of_size_N(lst, index=0, current=[], N=2):
    sub = []
    for subset in itertools.combinations(lst, N):
        if subset:
            sub.append(list(subset))
        
    return sub


def sublists_till_size_N(lst, index=0, current=[], N=2):
    sub = []
    for L in range(N+1):
        for subset in itertools.combinations(lst, L):
            if subset:
                sub.append(list(subset))
        
    return sub


########################################### COUNTERFACTUAL UTILITY FUNCTIONS ###########################################

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
    diff = (x_counterfactual-x)**2
    distance = 0
    for i in diff:
        distance += i
    return distance


########################################### CAUSALITY UTILITY FUNCTIONS ###########################################
#These two methods have been taken from a sample notebook on causal-learn and is used in generating causal structures

def make_graph(adjacency_matrix, labels=None):
    idx = np.abs(adjacency_matrix) > 0.01
    dirs = np.where(idx)
    d = graphviz.Digraph(engine='dot')
    names = labels if labels else [f'x{i}' for i in range(len(adjacency_matrix))]
    for name in names:
        d.node(name)
    for to, from_, coef in zip(dirs[0], dirs[1], adjacency_matrix[idx]):
        d.edge(names[from_], names[to], label=str(coef))
    return d

def str_to_dot(string):
    '''
    Converts input string from graphviz library to valid DOT graph format.
    '''
    graph = string.strip().replace('\n', ';').replace('\t','')
    graph = graph[:9] + graph[10:-2] + graph[-1] # Removing unnecessary characters from string
    return graph
        

########################################### DATASET UTILITY FUNCTIONS ###########################################

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

#Just a way to write the effects due to changes in counterfactual in words - here we ignore if features are unchanged
def denormalized_effects(x, counterfactual, data, col_name, denorm_dict, normalization):
    op_lst = reverse_normalize(x.copy(), data, col_name, denorm_dict, normalization) - reverse_normalize(counterfactual.copy(), data, col_name, denorm_dict, normalization)
    st = ""
    for i, val in enumerate(op_lst):
        if val>0:
            st = st + data.columns[i] + " increased by " + str(abs(val)) + ", "
        elif val<0:
            st = st + data.columns[i] + " decreased by " + str(abs(val)) + ", "
    st = st.rstrip(', ')
    return st


#Just a way to write the changes to be made by user in words - here we consider if features are unchanged
def denormalized_changes(x, counterfactual, b, data, col_name, denorm_dict, normalization):
    op_lst = reverse_normalize(x.copy(), data, col_name, denorm_dict, normalization) - reverse_normalize(counterfactual.copy(), data, col_name, denorm_dict, normalization)
    st = ""
    for _, val in enumerate(b):
        if op_lst[val-1]>0:
            st = st + data.columns[val-1] + " increased by " + str(abs(op_lst[val-1])) + ", "
        elif op_lst[val-1]<0:
            st = st + data.columns[val-1] + " decreased by " + str(abs(op_lst[val-1])) + ", "
        elif op_lst[val-1]==0:
            st = st + data.columns[val-1] + " unchanged" + ", "
    st = st.rstrip(', ')
    return st


########################################### DATA NORMALIZATION FUNCTIONS ###########################################

def normalize(dataset, col_name, normalization):
    if normalization == "minmax":
        for i in dataset.columns:
            if i!=col_name:
                dataset[i] = (dataset[i]-dataset[i].min())/(dataset[i].max()-dataset[i].min())
    elif normalization == "zscore":
        for i in dataset.columns:
            if i!=col_name:
                dataset[i] = (dataset[i]-dataset[i].mean())/(dataset[i].std())
    return dataset

def reverse_normalize(x, data, col_name, denorm_dict, normalization):
    if normalization == "minmax":
        for i, var in enumerate(data.columns):
            if i!=col_name:
                x[i]=x[i]*(denorm_dict[var][1] - denorm_dict[var][0]) + denorm_dict[var][0]
    elif normalization == "zscore":
        for i, var in enumerate(data.columns):
            if i!=col_name:
                x[i]=x[i]*denorm_dict[var][1] + denorm_dict[var][0]
    return x

# #just feature scaling
# def normalize(dataset, col_name):
#     for i in dataset.columns:
#         if i!=col_name:
#             dataset[i] = dataset[i]/1000
#     return dataset



########################################### MAIN FUNCTIONS ###########################################

''' Given: 
        gap_between_prints: Used for log printing purposes - to decide frequency of prints (int)
        ft: List of features to change (list) (indexing from 1)
        data: Normalized Dataset (dataframe)
        N: Number of features (includes Y as a column) - always use N-1 in your calculations (int)
        model: The black box model (.pth file)
        lr: Learning rate (float)
        max_iter: Maximum Iterations for pytorch optimization (int)
        _lambda: Lambda to decide the weightage to L2 loss and classification loss in Wachter Loss equation (float)
        x: Input from user (np.float32 numpy array)
        model_scm: SCM learnt from data (Lingam model)
        labels: Labels for SCM nodes - useful for Lingam inference (list) (column names)
        graph_dot: Useful for Lingam inference
        max_number_of_permutations: Maximum number of permutations considered for any intervention state
        history_of_causal_effect: Dictionary to store previous history of causal inferences in order to speed up causal inference estimation
    Output: Counterfactual using all the features in ft and cost of generating counterfactual
'''
def gen_cf(gap_between_prints, ft, data, N, model, lr, max_iter, _lambda, x, model_scm, labels, graph_dot, max_number_of_permutations, history_of_causal_effect):
    obj = SeventhMethod.SeventhMethod(data, model, model_scm, labels, graph_dot, history_of_causal_effect)
    min_cost = float('inf')
    best_counterfactual = None
    y_hat_best_counterfactual = None
    
    if len(ft)==1: #That is, only single element sent -> No need to permute or find order twice (once to find best_order as below in else condition, and another when finding with the best order)
        logger.debug("\n{}: Single feature hence no permutation needed.".format(datetime.datetime.now()))
        best_counterfactual = x.copy() #We take copy since we do not want to destroy the original
        best_counterfactual, min_cost, y_hat_best_counterfactual = obj.generate_counterfactuals(x.copy(), best_counterfactual, features_to_vary = ft, N=N, learning_rate=lr, max_iter=max_iter, _lambda=_lambda, target = 1, gap_between_prints = gap_between_prints)
        
    else:
        max_number_of_permutations = min(max_number_of_permutations, math.factorial(len(ft))) #max_number_of_permutations cannot be more than number of permutations
        list_of_permutations = []
        while(len(list_of_permutations)<max_number_of_permutations):
            permutation = ft.copy()
            random.shuffle(permutation)
            logger.debug("{}: Permutation {} from ft {} => Is it absent in list_of_permutations? {}.".format(datetime.datetime.now(), permutation, ft, permutation not in list_of_permutations))
            if permutation is not None and permutation not in list_of_permutations:
                list_of_permutations.append(permutation)
        #NOTE: The previous methodology of generating all possible permutations and then choosing a subsample fails when it comes to huge number of features. Hence we take this optimized approach.

        logger.debug("\n")
        #Ordering of actions for any permutation of the features: perm[0] -> perm[1] -> perm[2] ->....
        for perm in list_of_permutations:
            cost = 0
            x_cf = x.copy() #We take a copy so original is not destroyed
            logger.debug("{}: Current permutation being checked = {}".format(datetime.datetime.now(), perm))
            for i in perm:
                logger.debug("{}: ********************************************************Generating Counterfactuals for permutation {} -> Currently checking for action {}********************************************************".format(datetime.datetime.now(), perm, i))
                x_cf, cost_ind, y_hat = obj.generate_counterfactuals(x.copy(), x_cf, features_to_vary = [i], N=N, learning_rate=lr, max_iter=max_iter, _lambda=_lambda, target = 1, gap_between_prints = gap_between_prints)
                if x_cf is None: #Failed to generate counterfactual -> Reached NaN values
                    break
                cost += cost_ind #classification cost of the current ordering of actions after each of their iterations: perm[0] -> perm[1] -> perm[2] ->....
            if x_cf is not None: #A valid counterfactual has been generated
                cost += _lambda*find_distance(x, x_cf) #Add the L1 loss at this instance - if cost_ind had L1 loss it would be finding L1 loss for perm[1], perm[2], ... wrt itself, not x
                logger.debug("{}: Cost of current permutation = {}".format(datetime.datetime.now(), cost))
                if cost<min_cost:
                    logger.debug("{}: Cost of current permutation is the new minimum! New minimum = {}".format(datetime.datetime.now(), cost))
                    min_cost = cost
                    best_counterfactual = x_cf
                    y_hat_best_counterfactual = y_hat

    if y_hat_best_counterfactual is None or y_hat_best_counterfactual<=0.5: #we failed to change the classification
        #For the best order, generate counterfactual (lowest cost)
        logger.debug("{}: ********************************************************Counterfactual generated = {} with classification cost = {} and y_hat = {}- DOES NOT PROVIDE RECOURSE! FAILED!********************************************************\n\n".format(datetime.datetime.now(), best_counterfactual, min_cost, y_hat_best_counterfactual))
        best_counterfactual = None
        min_cost = -1
    else:
        #For the best order, generate counterfactual (lowest cost)
        logger.debug("{}: ********************************************************Best counterfactual = {} with classification cost = {} and y_hat = {}********************************************************\n\n".format(datetime.datetime.now(), best_counterfactual, min_cost, y_hat_best_counterfactual))

    return best_counterfactual, min_cost, y_hat_best_counterfactual





''' Given: 
        gap_between_prints: Used for log printing purposes - to decide frequency of prints (int)
        dataset: Path of dataset (string)
        col_name: Name of output column (string)
        data: Normalized dataset (dataframe) with output column
        denorm_dict: Dictionary used to store relevant parameters for denormalizing input data in 'data' variable
        N: Number of features (includes Y as a column) - always use N-1 in your calculations (int)
        model: The black box model (.pth file)
        lr: Learning rate (float)
        max_iter: Maximum Iterations for pytorch optimization (int)
        _lambda: Lambda to decide the weightage to L2 loss and classification loss in Wachter Loss equation (float)
        x: Input from user (np.float32 numpy array)
        model_scm: SCM learnt from data (Lingam model)
        labels: Labels for SCM nodes - useful for Lingam inference (list) (column names)
        T: Given time horizon aka user-input bound for number of comparisons - this plays a role in our duelling bandit algorithm (See IF1 algorithm) (int)
        fixed_ft: Features which are fixed (list - indexing from 1)
        user_choice_subset: User subset choice - used for automated run purposes (indexing from 1)
        normalization: Type of normalization used
        max_number_of_permutations: Maximum number of permutations considered for any intervention state
    Output: Final Counterfactual, it's cost, total number of iterations needed, time taken to run, final subset chosen by algorithm as most suitable to users and y_hat for that counterfactual
'''
def run(gap_between_prints, dataset, col_name, data, denorm_dict, N, model, lr, _lambda, max_iter, x, model_scm, labels, T, fixed_ft = None, user_choice_subset = None, normalization = "minmax", max_number_of_permutations = 3):
    time_taken = 0.0
    dataset_name = os.path.splitext(os.path.basename(dataset))[0] #extract the name of the dataset used from the path given in dataset (no extension)
    time_start = time()

    if fixed_ft is None:
        fixed_ft = [int(item) for item in input("Enter the features which you do not want altered: ").split()] #Features which cannot be modified

    sblist = [] 
    for i in range(1, N):
        if i not in fixed_ft: #do not include fixed features
            sblist.append(i)
    suborders = sublists(sblist) #create all possible subsets of features

    dict_of_subset = {} #used to store tuples for each subset: (subset ID, P(b_hat>b), Confidence Interval, number of comparisons made for this pair)

    W = suborders.copy()
    data = data.copy().drop([col_name], axis=1)
    
    logger.debug("{}: W = {}".format(datetime.datetime.now(), W))

    K = len(W)
    delta = 1/(T*(K**2))
    b_hat = random.choice(W)
    W.remove(b_hat)

    logger.debug("{}: K = {}".format(datetime.datetime.now(), K))
    logger.debug("{}: 1-delta = {}".format(datetime.datetime.now(), 1-delta))

    #This loop is for creating the dictionary => We use tuple(subset) since lists cannot be used as keys. We need to use tuples instead.
    for i, subset in enumerate(W):
        #P(a>b) = e^-(len(a))/(e^-(len(a))+e^-(len(b))) => This is the Bradley Terry model we choose with u = e^-len(a), since we want to restrict on the number of features changed, thus if len(a) increase, e^(-len(a)) decreases
        dict_of_subset[tuple(subset)] = (i, np.exp(-len(b_hat))/(np.exp(-len(subset)) + np.exp(-len(b_hat))), np.sqrt(np.log(1/delta)), 1) #(subset ID, P(b_hat>b), Confidence Interval, number of comparisons made for this pair)

    c = 0.05 #This is the value we update P(b_hat,b) with
    c_original = c
    T_hat = 0 #Number of computations
    logger.debug("{}: Original data point is {}.".format(datetime.datetime.now(), x))
    
    final_candidate = None
    graph_dot = make_graph(model_scm.adjacency_matrix_, labels=labels)
    

    #Render the SCM graph for understanding
    rand = np.random.randint(low = 1, high = 20000)
    scm_test = 'SCM{}.pdf'.format(rand)
    scm_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../logs/scm_dump", scm_test)
    prev_files = glob.glob(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../logs/scm_dump/*"))
    for f in prev_files:
        os.remove(f)
    logger.debug("{}: Generated SCM can be found at {}.".format(datetime.datetime.now(), scm_path))
    make_dot(model_scm.adjacency_matrix_, labels=labels).render(engine='dot', view=False, outfile=scm_path)

    W_copy = W.copy()

    history_of_causal_effect = {} #This dictionary will be used to store previously tracked causal history

    while(len(W_copy)): #do while W is not empty
        if b_hat in W_copy:
            W_copy.remove(b_hat)
        next_candidate = None
        logger.debug("\n\n{}: ########################################################Generating Counterfactuals for b_hat {}########################################################".format(datetime.datetime.now(), b_hat))
        b_hat_counterfactual, _, _ = gen_cf(
                                        gap_between_prints = gap_between_prints,
                                        ft = b_hat,
                                        data = data,
                                        N = N,
                                        model = model,
                                        lr = lr,
                                        max_iter = max_iter,
                                        _lambda = _lambda,
                                        x = x.copy(),
                                        model_scm = model_scm,
                                        labels = labels,
                                        graph_dot = graph_dot,
                                        max_number_of_permutations = max_number_of_permutations,
                                        history_of_causal_effect = history_of_causal_effect
                                    )

        if b_hat_counterfactual is None: #If we failed to generate counterfactual, we look for another b_hat
            logger.debug("{}: b_hat {} needs to be changed due to unsuccessful counterfactual.".format(datetime.datetime.now(), b_hat))
            b_hat = random.choice(W_copy)
            W_copy.remove(b_hat)
            del dict_of_subset[tuple(b_hat)]
            if len(W_copy)==0: #The b_hat chosen from previous round was faulty - now we have no solution
                final_candidate = None
                break
            for i, subset in enumerate(W_copy): #Recompute P and C for the new b_hat
                dict_of_subset[tuple(subset)] = (i, np.exp(-len(b_hat))/(np.exp(-len(subset)) + np.exp(-len(b_hat))), np.sqrt(np.log(1/delta)), 1)
        
        else: #b_hat counterfactual found
            W_second_copy = W_copy.copy() #If we use W_copy, then due to removal of b, we can skip on some elements by mistake (Index 1 becomes 0 if W_copy[0] is removed -> If we use W_copy, then it would refer to W_copy[2] in original which becomes W_copy[0], thereby skipping on W_copy[1] which shoudl ahve been next)
            for b in W_second_copy:
                logger.debug("\n{}: Currently W = {}".format(datetime.datetime.now(), W_copy))
                logger.debug("\n\n{}: ########################################################Generating Counterfactuals for current b {}########################################################".format(datetime.datetime.now(), b))

                b_counterfactual, _, _ = gen_cf(
                                            gap_between_prints = gap_between_prints,
                                            ft = b,
                                            data = data,
                                            N = N,
                                            model = model,
                                            lr = lr,
                                            max_iter = max_iter,
                                            _lambda = _lambda,
                                            x = x.copy(),
                                            model_scm = model_scm,
                                            labels = labels,
                                            graph_dot = graph_dot,
                                            max_number_of_permutations = max_number_of_permutations,
                                            history_of_causal_effect = history_of_causal_effect
                                        )
                
                if b_counterfactual is None: #If we failed to generate counterfactual, we look for next b
                    logger.debug("{}: Remove b due to not being able to generate counterfactual. = {}".format(datetime.datetime.now(), b))
                    W_copy.remove(b)
                    del dict_of_subset[tuple(b)]

                else:
                    #used for printing purposes -> This is for effects reflected overall including causal effects
                    st_hat = denormalized_effects(x.copy(), b_hat_counterfactual.copy(), data, col_name, denorm_dict, normalization)
                    st = denormalized_effects(x.copy(), b_counterfactual.copy(), data, col_name, denorm_dict, normalization)
                    
                    #used for printing purposes -> This is for actual changes to be done by user
                    st_act_ft_hat = denormalized_changes(x.copy(), b_hat_counterfactual.copy(), b_hat, data, col_name, denorm_dict, normalization)
                    st_act_ft = denormalized_changes(x.copy(), b_counterfactual.copy(), b, data, col_name, denorm_dict, normalization)

                    if user_choice_subset is None: #When we want to make an interactive input about best subset
                        print("\nEnter the counterfactual you like more: ")
                        print("Option 1: {} with {} changes (Effects: {}).".format(st_act_ft_hat, find_change_count(x, b_hat_counterfactual), st_hat))
                        print("Option 2: {} with {} changes (Effects: {}).".format(st_act_ft,  find_change_count(x, b_counterfactual), st))
                        ch = input("Enter your choice: ")


                        #P(b_hat, b) updated => We reassign each dictionary item as a new tuple since tuples are not modifiable
                        logger.info("{}: Option 1: {} with {} changes (Effects: {}).".format(datetime.datetime.now(), st_act_ft_hat, find_change_count(x, b_hat_counterfactual), st_hat))
                        logger.info("{}: Option 2: {} with {} changes (Effects: {}).".format(datetime.datetime.now(), st_act_ft,  find_change_count(x, b_counterfactual), st))
                        logger.debug("{}: P(b_hat>b) = {}".format(datetime.datetime.now(), dict_of_subset[tuple(b)][1]))
                        logger.debug("{}: C(b_hat, b) = {}".format(datetime.datetime.now(), dict_of_subset[tuple(b)][2]))
                        
                        if ch=='1' or ch=='2':
                            ch = int(ch)
                        else:
                            ch = 1 #default is b_hat

                        logger.info("{}: User choice is Option {}.".format(datetime.datetime.now(), ch))
                    
                    else: #For when sending the choice directly (automated call)
                        score1 = 0
                        score2 = 0
                        for i in user_choice_subset: #if any feature in user's preferred subset lies in any of the options (b_hat and b), then their score increases
                            if i in b_hat:
                                score1 = score1 + 1
                            if i in b:
                                score2 = score2 + 1
                        if score1>score2: #Choose the one which matches user's preferred subset most
                            ch=1
                        elif score1<score2:
                            ch=2
                        else: #in case of match, choose the shorter option
                            if len(b_hat)>len(b):
                                ch = 2
                            else:
                                ch = 1
                        logger.info("{}: Logically chosen choice is Option {}.".format(datetime.datetime.now(), ch))

                    
                    if dict_of_subset[tuple(b)][1] >=0.5: #P(b_hat>b) >=0.5
                        if ch == 1: #User agrees with our prediction
                            logger.debug("{}: User agrees with our choice CP1".format(datetime.datetime.now()))
                            dict_of_subset[tuple(b)] = (dict_of_subset[tuple(b)][0], 
                                                        min(dict_of_subset[tuple(b)][1] + c, 1), 
                                                        np.sqrt(np.log(1/delta)/(dict_of_subset[tuple(b)][3]+1)), #C(b_hat, b) updated
                                                        dict_of_subset[tuple(b)][3]+1) #Number of comparisons for (b_hat, b) updated
                        else:
                            logger.debug("{}: User does not agree with our choice CP2".format(datetime.datetime.now()))
                            dict_of_subset[tuple(b)] =  (dict_of_subset[tuple(b)][0],
                                                        min(dict_of_subset[tuple(b)][1] - c, 0.5), #drop down the probability to at max 0.5
                                                        np.sqrt(np.log(1/delta)/(dict_of_subset[tuple(b)][3]+1)),
                                                        dict_of_subset[tuple(b)][3]+1)
                            
                    # elif dict_of_subset[tuple(b)][1] <0.5 and dict_of_subset[tuple(b)][1] >=0: #P(b_hat>b) <0.5
                    else: #P(b_hat>b) <0.5
                        if ch == 1:
                            logger.debug("{}: User does not agree with our choice CP3".format(datetime.datetime.now()))
                            dict_of_subset[tuple(b)] =  (dict_of_subset[tuple(b)][0],
                                                        max(dict_of_subset[tuple(b)][1] + c, 0.5), #bump up the probability at least to 0.5
                                                        np.sqrt(np.log(1/delta)/(dict_of_subset[tuple(b)][3]+1)),
                                                        dict_of_subset[tuple(b)][3]+1)
                        else: #User agrees with our prediction
                            logger.debug("{}: User agrees with our choice CP4".format(datetime.datetime.now()))
                            dict_of_subset[tuple(b)] = (dict_of_subset[tuple(b)][0],
                                                        max(dict_of_subset[tuple(b)][1] - c, 0),
                                                        np.sqrt(np.log(1/delta)/(dict_of_subset[tuple(b)][3]+1)),
                                                        dict_of_subset[tuple(b)][3]+1)
                    


                    T_hat = T_hat + 1 #Keep track of all comparisons
                    logger.debug("{}: T_hat = {}".format(datetime.datetime.now(), T_hat))
                    logger.debug("{}: b_hat = {}".format(datetime.datetime.now(), b_hat))
                    logger.debug("{}: b = {}".format(datetime.datetime.now(), b))
                    logger.debug("{}: P(b_hat>b) after change = {}".format(datetime.datetime.now(), dict_of_subset[tuple(b)][1]))
                    logger.debug("{}: C(b_hat>b) after change = {}".format(datetime.datetime.now(), dict_of_subset[tuple(b)][2]))
                    logger.debug("{}: t(b_hat>b) after change = {}".format(datetime.datetime.now(), dict_of_subset[tuple(b)][3]))

                    #PRUNING
                    if dict_of_subset[tuple(b)][1] > 0.5 and dict_of_subset[tuple(b)][2] >= 1 - delta: #We get an inferior bandit to b_hat with confidence > 1-delta => DISCARD
                        logger.debug("{}: Remove b = {}".format(datetime.datetime.now(), b))
                        W_copy.remove(b)
                        del dict_of_subset[tuple(b)]
                    #CHOOSING THE NEXT CANDIDATE
                    elif dict_of_subset[tuple(b)][1] < 0.5 and dict_of_subset[tuple(b)][2] >= 1 - delta: #b is more preferred than b_hat with confidence > 1-delta
                        if next_candidate is None or (next_candidate!=b_hat and dict_of_subset[tuple(b)][1] < dict_of_subset[tuple(next_candidate)][1]): #P(b_hat>b) < P(b_hat>nc) i.e b is stronger than nc => We get a superior bandit
                            next_candidate = b
                    
                    if next_candidate is not None and next_candidate!=b_hat:
                        logger.debug("{}: Next candidate = {} with P(b_hat>nc) = {}".format(datetime.datetime.now(), next_candidate, dict_of_subset[tuple(next_candidate)][1]))
                    else:
                        logger.debug("{}: Next candidate = {}".format(datetime.datetime.now(), next_candidate))



            logger.debug("{}: W after loop = {}".format(datetime.datetime.now(), W_copy))
            logger.debug("{}: Final candidate = {}".format(datetime.datetime.now(), final_candidate))

            if len(W_copy)==0:
                final_candidate = b_hat
                break
            elif next_candidate is None:
                logger.debug("{}: Next candidate b_hat unchanged = {}".format(datetime.datetime.now(), b))
                next_candidate = b_hat

            #choose the new candidate
            if set(next_candidate) != set(b_hat):
                c = c_original #if we don't get the same candidate, then reset the value to modify comparison probability
                W_copy.remove(next_candidate)
                del dict_of_subset[tuple(next_candidate)]
                if len(W_copy)==0:
                    final_candidate = b_hat
                    break
                b_hat = next_candidate
                for i, subset in enumerate(W_copy): #Recompute P and C for the new b_hat
                    dict_of_subset[tuple(subset)] = (i, np.exp(-len(b_hat))/(np.exp(-len(subset)) + np.exp(-len(b_hat))), np.sqrt(np.log(1/delta)), 1)
            else:
                c = c*2 #if we get the same candidate, then next time the comparison probability are modified more aggressively
        
    if final_candidate is not None:
        logger.debug("{}: Final subset chosen by user = {}".format(datetime.datetime.now(), final_candidate))
        final_counterfactual, cost, y_hat = gen_cf(
                                        gap_between_prints = gap_between_prints,
                                        ft = final_candidate,
                                        data = data,
                                        N = N,
                                        model = model,
                                        lr = lr,
                                        max_iter = max_iter,
                                        _lambda = _lambda,
                                        x = x,
                                        model_scm = model_scm,
                                        labels = labels,
                                        graph_dot = graph_dot,
                                        max_number_of_permutations = max_number_of_permutations,
                                        history_of_causal_effect = history_of_causal_effect
                                    )
        if final_counterfactual is None:
            logger.debug("{}: No solution was found!".format(datetime.datetime.now()))
            final_counterfactual = None
            cost = -1
            T_hat = -1
            y_hat = -1
        else:
            #used for printing purposes -> This is for effects of changes
            final_st = denormalized_effects(x.copy(), final_counterfactual.copy(), data, col_name, denorm_dict, normalization)
            #used for printing purposes -> This is for actual changes to be done by user
            final_st_act = denormalized_changes(x.copy(), final_counterfactual.copy(), final_candidate, data, col_name, denorm_dict, normalization)
            logger.info("{}: Final changes chosen by user are {} (Effects: {})".format(datetime.datetime.now(), final_st_act, final_st))

    else:
        logger.debug("{}: No solution was found!".format(datetime.datetime.now()))
        final_counterfactual = None
        cost = -1
        T_hat = -1
        y_hat = -1


    time_end = time()
    time_taken = time_end-time_start
    
    if final_counterfactual is not None:
        #This is a model-independent check for what the output data should have been and is not model dependent
        y_act = find_actual_value(final_counterfactual, dataset_name)
        if y_act!=-1:
            logger.info("{}: Actual y for this counterfactual is = {}".format(datetime.datetime.now(), y_act))
        else:
            logger.info("{}: Actual output for this counterfactual is not possible since this is a real world dataset.".format(datetime.datetime.now()))
    logger.info("-------------------------------------------------------------------------------------------------------\n")

    return final_counterfactual, cost, T_hat, time_taken, final_candidate, y_hat
    


''' Given:
        dataset: Path of dataset (string)
        T: Given time horizon aka user-input bound for number of comparisons - this plays a role in our duelling bandit algorithm (See IF1 algorithm) (int)
        index_x: In case user chooses a specific index, by default it is -1 indicating that use any random index with negative classification
        lr: Learning rate (float)
        _lambda: Lambda to decide the weightage to L2 loss and classification loss in Wachter Loss equation (float)
        max_iter: Maximum Iterations for pytorch optimization (int)
        gap_between_prints: Used for log printing purposes - to decide frequency of prints (int)
        size_of_dataset: Used to indicate how many users to consider in your dataset with negative outcomes (int)
        seed: Used for reproducability reasons (int)
        type_of_model: Used to indicate what black box model to use in an automated scenario (int)
        fixed_ft: Features which are fixed (list - indexing from 1)
        user_choice_subset: User subset choice - used for automated run purposes (indexing from 1)
        normalization: What type of normalization to use
        max_number_of_permutations: Maximum number of permutations considered for any intervention state
    Output: Number of features changed in the best counterfactual, it's cost, time taken to run, total number of iterations needed, and final subset chosen by algorithm as most suitable to users
'''
def main(dataset, T=100, index_x=-1,  lr=0.01, _lambda=0.1, max_iter=100, gap_between_prints=80, size_of_dataset=5000, seed=1, type_of_model = -1, fixed_ft=None, user_choice_subset=None, normalization = "minmax", max_number_of_permutations = 3):
    #Input section
    data=pd.read_csv(dataset)
    data = data.dropna()
    if(size_of_dataset>data.shape[0]):
        size_of_dataset = data.shape[0]
    data = data.head(size_of_dataset) #size of dataset to be used
    
    dataset_name = os.path.splitext(os.path.basename(dataset))[0] #extract the name of the dataset used from the path given in dataset (no extension)
    N=len(data.columns)
    
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

    #as of now assume data is not normalized by traditional techniques
    data = normalize(data, col_name, normalization)

    #SCM Configuration
    if(dataset_name[:-1]=="dataset" and int(dataset_name[-1])<4): #dataset 1-3 are not causal
        pass
    else:
        logger.info('{}: Creating Causal Model....'.format(datetime.datetime.now()))
        data_copy = data.copy().drop([col_name], axis=1).dropna()
        labels = [f'{col}' for i, col in enumerate(data_copy.columns)]
        model_scm = lingam.ICALiNGAM()
        # model = lingam.RCD()
        model_scm.fit(data_copy)
        logger.info('{}: Causal Model created.'.format(datetime.datetime.now()))
        make_dot(model_scm.adjacency_matrix_, labels=labels).render('.\My Method\logs\SCM.gv', view=False, cleanup=True)



    #save the normalized data to check
    data.to_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/normalized", dataset_name + "_normalized.csv"), index=False)
    
    #If x given by user
    if(index_x!=-1):
        x = np.float32(np.reshape(data.drop([col_name], axis=1).iloc[index_x].tolist(), (1,N-1))[0]) #conversion to float32 is needed so that converting to Type32 tensor doesnt give unexpected errors
        logger.info('{}: Chosen data point is {} at index {} with {} = {}.'.format(datetime.datetime.now(), reverse_normalize(x.copy(), data.copy().drop([col_name], axis=1), col_name, denorm_dict, normalization), index_x, col_name, data.iloc[index_x][col_name]))
        logger.info('{}: Chosen data point normalized is {}.'.format(datetime.datetime.now(), x))
    #If x is not given by user
    else:
        index_x = random.choice(data.index[data[col_name]==0].tolist())
        x = np.float32(np.reshape(data.drop([col_name], axis=1).iloc[index_x].tolist(), (1,N-1))[0])
        logger.info('{}: Chosen data point is {} at a randomized index {} with {} = {}.'.format(datetime.datetime.now(), reverse_normalize(x.copy(), data.copy().drop([col_name], axis=1), col_name, denorm_dict, normalization), index_x, col_name, data.iloc[index_x][col_name]))
        logger.info('{}: Chosen data point normalized is {}.'.format(datetime.datetime.now(), x))
        

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
    elif(ch==2):
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
    elif(ch==3):
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
    elif(ch==4): #default is taken as MLP model
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
    model.eval() #no more training on this model

    #Output section
    best_counterfactual, cost, T_hat, time_taken, final_candidate, y_hat = run(
                                                                        gap_between_prints = gap_between_prints,
                                                                        dataset = dataset,
                                                                        col_name = col_name,
                                                                        data = data,
                                                                        denorm_dict = denorm_dict,
                                                                        N = N,
                                                                        model = model,
                                                                        lr = lr,
                                                                        _lambda = _lambda,
                                                                        max_iter = max_iter,
                                                                        x = x,
                                                                        model_scm = model_scm,
                                                                        labels = labels,
                                                                        T = T,
                                                                        fixed_ft = fixed_ft,
                                                                        user_choice_subset = user_choice_subset,
                                                                        normalization = normalization,
                                                                        max_number_of_permutations = max_number_of_permutations
                                                                    )

    if(best_counterfactual is not None and y_hat>0.5):
        number_of_features_changed = len(final_candidate)
        cost_of_counterfactual = cost
        logger.info('{}: Original data point is {}'.format(datetime.datetime.now(), x))
        logger.info('{}: The counterfactual is given by {} with number of features changed = {}, and L1 cost = {}'.format(datetime.datetime.now(), best_counterfactual, number_of_features_changed, cost_of_counterfactual))
        logger.info('{}: Time taken {} seconds'.format(datetime.datetime.now(), time_taken))
        logger.info('{}: Prediction = {}'.format(datetime.datetime.now(), y_hat))
        # logger.info('{}: Model output on Counterfactual is {}'.format(datetime.datetime.now(), avg_violation))
    else:
        logger.info('{}: Could not find a counterfactual in the given iterations.'.format(datetime.datetime.now()))
        x_counterfactual = None
        number_of_features_changed = -1
        cost_of_counterfactual = -1

    return number_of_features_changed, cost_of_counterfactual, time_taken, T_hat, final_candidate




########################################### FOR RUNNING THIS SINGLE SCRIPT ###########################################

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
        
    index_x = 9 #1 for HELOC, 2 for dataset3, 4 for dataset4, 9 for give_me_credit
    # index_x = -1
    learning_rate = 1e-1
    _lambda = 0.7
    max_iterations = 5
    gap_between_prints = 1
    size_of_dataset = 200
    seed = 3
    T = 10
    fixed_ft = [2, 5, 7]
    # fixed_ft = None
    user_choice_subset = [2, 4, 9]
    # user_choice_subset = None
    type_of_model = 3
    normalization="zscore"
    max_number_of_permutations = 4

    main(dataset = dataset,
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




