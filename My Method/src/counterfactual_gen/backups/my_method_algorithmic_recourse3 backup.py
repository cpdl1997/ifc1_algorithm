"""
This is an implementation which uses duelling bandits to generate counterfactuals

"""


import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
import pandas as pd
import numpy as np
# np.set_printoptions(precision=3) #only 3 decimal digits will be shown for any numpy array
from dataset_gen import dataset_generator as dg
# from model import custom_sequential, mlp_classifier
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../"))
from algorithms import MananWithCausalityCF, WachterCF
from scm import SCM, Node
import util.QueueSet as QS
import util.PrioritySet as PS
import datetime
import random
import copy
import itertools
from time import time


import logging
log_file_name = "Log_my_method_algorithmic_recourse3.log"
log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../logs", log_file_name)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(log_path, mode='w')
fh.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)
logger.addHandler(fh)

import warnings
warnings.filterwarnings('ignore')


import torch
import torch.nn as nn
import torch.nn.functional as F


def sublists(lst, index=0, current=[]):
    sub = []
    for L in range(len(lst)+1):
        for subset in itertools.combinations(lst, L):
            if subset:
                sub.append(list(subset))
        
    return sub

def sublists_of_size_N(lst, index=0, current=[], N=2):
    sub = []
    for subset in itertools.combinations(lst, N):
        if subset:
            sub.append(list(subset))
        
    return sub

#Given: List of features ft (list), dataset data (dataframe), model (model), input x (list)
#Output: Counterfactual using all the features in ft
def gen_cf(gap_between_prints, ft, data, N, model, lr, max_iter, l, x, scm):
    # obj = MananWithCausalityCF.MananWithCausalityCF(data, model, scm)
    # x_cf = obj.generate_counterfactuals(x, features_to_vary = ft, N=N, learning_rate=lr, max_iter=max_iter, _lambda=l, target = 1, gap_between_prints = gap_between_prints) #to be implemented
    obj = WachterCF.WachterCF(data, model)
    x_cf = obj.generate_counterfactuals(x, features_to_vary = ft, N=N, learning_rate=lr, max_iter=max_iter, _lambda=l, target = 1, gap_between_prints = gap_between_prints) #to be implemented
    
    return x_cf




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


#prints the SCM
def print_SCM(node: Node.Node, queue:PS.PrioritySet):
    if node.children is None:
        logger.debug("{}: At level {} we have feature {} with value = {} with no children.".format(datetime.datetime.now(), node.level, node.name, node.value))
        return
    logger.debug("{}: At level {} we have feature {} with value = {} with children: {}".format(datetime.datetime.now(), node.level, node.name, node.value, [i.name for i in node.children ]))
    for i in node.children:
        queue.add(i, i.level)
    while (queue.isempty()==False):
        nextNode = queue.pop()
        print_SCM(nextNode, queue)

#fills the SCM
def fill_SCM(node: Node.Node, input, queue:PS):
    node.value = input[node.index]
    if node.children is None:
        return
    for i in node.children:
        queue.add(i, i.level)
    while not queue.isempty():
        nextNode = queue.pop()
        fill_SCM(nextNode, input, queue)
        

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
# def normalize(dataset):
#     for i in dataset.columns:
#         dataset[i] = (dataset[i]-dataset[i].min())/(dataset[i].max()-dataset[i].min())
#     return dataset

#z-score normalisation
def normalize(dataset, col_name):
    for i in dataset.columns:
        if i!=col_name:
            dataset[i] = (dataset[i]-dataset[i].mean())/(dataset[i].std())
    return dataset
def reverse_normalize(x, data, denorm_dict):
    for i, var in enumerate(data.columns):
        x[i]=x[i]*denorm_dict[var][1] + denorm_dict[var][1]
    return x

# #just feature scaling
# def normalize(dataset, col_name):
#     for i in dataset.columns:
#         if i!=col_name:
#             dataset[i] = dataset[i]/1000
#     return dataset




#Given: Preference order pref_order (list), dataset df (dataframe), number of features N (int), model (model), input x (list)
#Output: Counterfactual which causes least number of violations
def run(gap_between_prints, dataset, col_name, data, denorm_dict, N, model, lr, l, max_iter, x, scm, T, fixed_ft = None, user_choice_subset = None):
    time_taken = 0.0
    dataset_name = os.path.splitext(os.path.basename(dataset))[0] #extract the name of the dataset used from the path given in dataset (no extension)
    time_start = time()
    suborders = sublists([i+1 for i in range(N-1)])
    dict_of_subset = {}
    if fixed_ft is None:
        fixed_ft = [int(item) for item in input("Enter the features which you do not want altered: ").split()] #Features which cannot be modified
    W = suborders.copy()
    data = data.copy().drop([col_name], axis=1)

    #This loop is just to remove the features
    for _, subset in enumerate(suborders):
        for i in fixed_ft: #The current subset has a feature which cannot be modified
            if i in subset and subset in W:
                W.remove(subset)
    
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
    logger.debug("{}: Original data point is {}.".format(datetime.datetime.now(), x, type(x)))
    
    final_candidate = None

    while(len(W)): #do while W is not empty
        if b_hat in W:
            W.remove(b_hat)
        next_candidate = None
        b_hat_counterfactual = gen_cf(gap_between_prints, b_hat, data, N, model, lr, max_iter, l, x, scm).detach().numpy()
        W_copy = W.copy()
        for b in W_copy:
            b_counterfactual = gen_cf(gap_between_prints, b, data, N, model, lr, max_iter, l, x, scm).detach().numpy()

            #used for printing purposes
            op_lst_hat = reverse_normalize(x.copy(), data, denorm_dict) - reverse_normalize(b_hat_counterfactual.copy(), data, denorm_dict)
            op_lst = reverse_normalize(x.copy(), data, denorm_dict) - reverse_normalize(b_counterfactual.copy(), data, denorm_dict)
            st_hat = ""
            st = ""
            for i, val in enumerate(op_lst_hat):
                if val>0:
                    st_hat = st_hat + data.columns[i] + " increased by " + str(abs(val)) + ", "
                elif val<0:
                    st_hat = st_hat + data.columns[i] + " decreased by " + str(abs(val)) + ", "
            for i, val in enumerate(op_lst):
                if val>0:
                    st = st + data.columns[i] + " increased by " + str(abs(val)) + ", "
                elif val<0:
                    st = st + data.columns[i] + " decreased by " + str(abs(val)) + ", "
            
            st_hat = st_hat.rstrip(', ')
            st = st.rstrip(', ')

            if user_choice_subset is None: #When we want to make an interactive input about best subset
                print("Enter the counterfactual you like more: ")
                print("Option 1: {} with {} changes.".format( st_hat, find_change_count(x, b_hat_counterfactual)))
                print("Option 2: {} with {} changes.".format(st, find_change_count(x, b_counterfactual)))
                ch = input("Enter your choice: ")


                #P(b_hat, b) updated => We reassign each dictionary item as a new tuple since tuples are not modifiable
                logger.debug("{}: Option 1 = {} with {} changes and features changed being {}.".format(datetime.datetime.now(), st_hat, find_change_count(x, b_hat_counterfactual), b_hat))
                logger.debug("{}: Option 2 = {} with {} changes and features changed being {}.".format(datetime.datetime.now(), st, find_change_count(x, b_counterfactual), b))
                logger.debug("{}: P(b_hat>b) = {}".format(datetime.datetime.now(), dict_of_subset[tuple(b)][1]))
                logger.debug("{}: C(b_hat, b) = {}".format(datetime.datetime.now(), dict_of_subset[tuple(b)][2]))
                
                if ch=='1' or ch=='2':
                    ch = int(ch)
                else:
                    ch = 1 #default is b_hat

                logger.debug("{}: User choice is Option {}.".format(datetime.datetime.now(), ch))
            
            else: #For when sending the choice directly
                score1 = 0
                score2 = 0
                for i in user_choice_subset:
                    if i in b_hat:
                        score1 = score1 + 1
                    if i in b:
                        score2 = score2 + 1
                if score1>score2:
                    ch=1
                elif score1<score2:
                    ch=2
                else: #in case of match, choose the shorter option
                    if len(b_hat)>len(b):
                        ch = 2
                    else:
                        ch = 1
                logger.debug("{}: Logically chosen choice is Option {}.".format(datetime.datetime.now(), ch))

            
            if dict_of_subset[tuple(b)][1] >=0.5:
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
            elif dict_of_subset[tuple(b)][1] <0.5 and dict_of_subset[tuple(b)][1] >=0:
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

            if next_candidate is not None and next_candidate!=b_hat:
                logger.debug("{}: Next candidate = {} with P(b_hat>nc) = {}".format(datetime.datetime.now(), next_candidate, dict_of_subset[tuple(next_candidate)][1]))
            else:
                logger.debug("{}: Next candidate = {}".format(datetime.datetime.now(), next_candidate))

            if dict_of_subset[tuple(b)][1] > 0.5 and dict_of_subset[tuple(b)][2] >= 1 - delta: #We get an inferior bandit with confidence > 1-delta
                logger.debug("{}: Remove b = {}".format(datetime.datetime.now(), b))
                W.remove(b)
                del dict_of_subset[tuple(b)]
            elif dict_of_subset[tuple(b)][1] < 0.5 and dict_of_subset[tuple(b)][2] >= 1 - delta:
                # if np.exp(-len(b))/(np.exp(-len(b)) + np.exp(-len(next_candidate))) > 0.5: #P(b>nc) > 0.5 => We get an superior bandit with confidence > 1-delta
                if next_candidate is None or (next_candidate!=b_hat and dict_of_subset[tuple(b)][1] < dict_of_subset[tuple(next_candidate)][1]): #P(b>nc) > 0.5 => We get an superior bandit with confidence > 1-delta
                    logger.debug("{}: Next candidate b_hat = {}".format(datetime.datetime.now(), b))
                    next_candidate = b

        logger.debug("{}: W after loop = {}".format(datetime.datetime.now(), W))
        logger.debug("{}: Final candidate = {}".format(datetime.datetime.now(), final_candidate))

        if len(W)==0:
            final_candidate = b_hat
            break
        elif next_candidate is None:
            logger.debug("{}: Next candidate b_hat unchanged = {}".format(datetime.datetime.now(), b))
            next_candidate = b_hat

        #choose the new candidate
        if set(next_candidate) != set(b_hat):
            c = c_original #if we don't get the same candidate, then reset the value to modify comparison probability
            W.remove(next_candidate)
            del dict_of_subset[tuple(next_candidate)]
            if len(W)==0:
                final_candidate = b_hat
                break
            b_hat = next_candidate
            for i, subset in enumerate(W): #Recompute P and C for the new b_hat
                dict_of_subset[tuple(subset)] = (i, np.exp(-len(b_hat))/(np.exp(-len(subset)) + np.exp(-len(b_hat))), np.sqrt(np.log(1/delta)), 1)
        else:
            c = c*2 #if we get the same candidate, then next time the comparison probability are modified more aggressively
        
            
    logger.debug("{}: Final subset chosen by user = {}".format(datetime.datetime.now(), final_candidate))
    time_end = time()
    time_taken += time_end-time_start
    
    final_counterfactual = gen_cf(gap_between_prints, final_candidate, data, N, model, lr, max_iter, l, x, scm).detach().numpy()
    #for printing the changes chosen by user
    op_diff = reverse_normalize(x.copy(), data, denorm_dict) - reverse_normalize(final_counterfactual.copy(), data, denorm_dict)
    final_st = ""
    for i, val in enumerate(op_diff):
        if val>0:
            final_st = final_st + data.columns[i] + " increased by " + str(abs(val)) + ", "
        elif val<0:
            final_st = final_st + data.columns[i] + " decreased by " + str(abs(val)) + ", "
    final_st = final_st.rstrip(', ')
    logger.info("{}: Final changes chosen by user are {}".format(datetime.datetime.now(), final_st))

    y_act = find_actual_value(final_counterfactual, dataset_name)
    if y_act!=-1:
        logger.info("{}: Actual y for this counterfactual is = {}".format(datetime.datetime.now(), y_act))
    else:
        logger.info("{}: Actual output for this counterfactual is not possible since this is a real world dataset.".format(datetime.datetime.now()))
    logger.info("-------------------------------------------------------------------------------------------------------\n")

    return final_counterfactual, T_hat, time_taken, final_candidate
    


#Given: The absolute path of the dataset to be used dataset (string), Index of preferred datapoint to find counterfactual of x (int - default value = -1)
#Output: Counterfactual which causes least number of violations
def main(dataset, T=100, index_x=-1,  lr=0.01, l=0.1, max_iter=100, gap_between_prints=80, size_of_dataset=5000, seed=1, type_of_model = -1, fixed_ft=None, user_choice_subset=None):
    #Input section
    data=pd.read_csv(dataset)
    if(size_of_dataset>data.shape[0]):
        size_of_dataset = data.shape[0]
    data = data.head(size_of_dataset) #size of dataset to be used
    
    dataset_name = os.path.splitext(os.path.basename(dataset))[0] #extract the name of the dataset used from the path given in dataset (no extension)
    N=len(data.columns)
    
    #output column name
    if(dataset_name[:-1]=="dataset"):
        col_name = 'Y'
        num = int(dataset_name[-1])
        root = Node.Node(feature_name='X1', feature_index=0,  feature_level=1)
        scm = SCM.SCM(input_ft=data.columns, root_node=root)
        #SCM config
        if(num<3):
            pass
        elif(num==4):
            scm.equations = torch.tensor([[1,0,0,0,0], [2,0,0,0,0], [3,0,0,0,0], [0,1,-2,0,0], [-1,0,2,0,0]], dtype=torch.float32, requires_grad=False)
            x2 = Node.Node(feature_value=0, feature_name='X2', feature_index=1, parent_list=[root], feature_level=2)
            x3 = Node.Node(feature_value=0, feature_name='X3', feature_index=2, parent_list=[root], feature_level=2)
            x4 = Node.Node(feature_value=0, feature_name='X4', feature_index=3, parent_list=[x2, x3], feature_level=3)
            x5 = Node.Node(feature_value=0, feature_name='X5', feature_index=4, parent_list=[root, x3], feature_level=3)
            x2.children = [x4]
            x3.children = [x4, x5]
            root.children = [x2, x3, x5]
        elif(num==5):
            pass #to be filled
        elif(num==6):
            pass #to be filled
        elif(num==7):
            pass #to be filled
    else:
        if(dataset_name=="adult"):
            #categorical_encoding
            col_name = 'income'
            root = Node.Node(feature_index=0)
            scm = SCM.SCM(data.columns, root)
            #scm to be incorporated
        elif(dataset_name=="compas"):
            #categorical_encoding
            col_name = 'score'
            root = Node.Node(feature_index=0)
            scm = SCM.SCM(data.columns, root)
            #scm to be incorporated
        elif(dataset_name=="give_me_some_credit"):
            col_name = 'SeriousDlqin2yrs'
            root = Node.Node(feature_index=0)
            scm = SCM.SCM(data.columns, root)
            #scm to be incorporated
        elif(dataset_name=="heloc"):
            col_name = 'RiskPerformance'
            root = Node.Node(feature_index=0)
            scm = SCM.SCM(data.columns, root)
            #scm to be incorporated
    
    #Keep track of original values so that we can denormalize
    #for z-score keep track of SD and Mean
    denorm_dict={}
    for var in data.columns:
        denorm_dict[var]=(data[var].mean(), data[var].std())

    #as of now assume data is not normalized by traditional techniques
    data = normalize(data, col_name)

    #save the normalized data to check
    data.to_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/normalized", dataset_name + "_normalized.csv"), index=False)
    

    if(index_x!=-1):
        x = np.float32(np.reshape(data.drop([col_name], axis=1).iloc[index_x].tolist(), (1,N-1))[0]) #conversion to float32 is needed so that converting to Type32 tensor doesnt give unexpected errors
        logger.info('{}: Chosen data point is {} at index {} with {} = {}.'.format(datetime.datetime.now(), x, index_x, col_name, data.iloc[index_x][col_name]))
    else:
        index_x = random.choice(data.index[data[col_name]==0].tolist())
        x = np.float32(np.reshape(data.drop([col_name], axis=1).iloc[index_x].tolist(), (1,N-1))[0])
        logger.info('{}: Chosen data point is {} at a randomized index {} with {} = {}.'.format(datetime.datetime.now(), x, index_x, col_name, data.iloc[index_x][col_name]))
    
    #fill the scm values using the data point and then print it
    logger.debug("{}: Filling SCM with values...".format(datetime.datetime.now()))
    queue = PS.PrioritySet()
    fill_SCM(scm.root, x, queue)
    queue.clear()
    print_SCM(scm.root, queue)

    unobservables = np.zeros(N-1)
    #first part of abduction-action-prediction involves abduction -> We are currently doing abduction
    for i in range(1, N):
        unobservables[i-1] = x[i-1]-np.dot(x, scm.equations[i-1].numpy())
    logger.debug('{}: Unobservables = {}'.format(datetime.datetime.now(), unobservables))
    scm.exogenous = torch.from_numpy(unobservables)

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

    #Output section
    best_counterfactual, T_hat, time_taken, final_candidate = run(gap_between_prints, dataset, col_name, data, denorm_dict, N, model,  lr, l, max_iter, x, scm, T, fixed_ft, user_choice_subset)

    if(model(torch.from_numpy(best_counterfactual))):
        number_of_features_changed = find_change_count(x, best_counterfactual)
        cost_of_counterfactual = find_distance(x, best_counterfactual)
        logger.info('{}: Original data point is {}'.format(datetime.datetime.now(), x))
        logger.info('{}: The counterfactual is given by {} with number of features changed = {}, and L1 cost = {}'.format(datetime.datetime.now(), best_counterfactual, number_of_features_changed, cost_of_counterfactual))
        logger.info('{}: Time taken {} seconds'.format(datetime.datetime.now(), time_taken))
        # logger.info('{}: Model output on Counterfactual is {}'.format(datetime.datetime.now(), avg_violation))
    else:
        logger.info('{}: Could not find a counterfactual in the given iterations.'.format(datetime.datetime.now()))
        x_counterfactual = None
        number_of_features_changed = -1
        cost_of_counterfactual = -1

    return number_of_features_changed, cost_of_counterfactual, time_taken, T_hat, final_candidate



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
        
    index_x = 90 #1 for HELOC, 2 for dataset3, 3 for dataset4
    learning_rate = 1e-3
    _lambda = 0.7
    max_iterations = 100
    gap_between_prints = 100
    size_of_dataset = 5000
    seed = 3
    T = 10
    fixed_ft = [1,2]
    user_choice_subset = [4,5]


    main(dataset = dataset,
         T = T,
         index_x = index_x,
         lr = learning_rate,
         l = _lambda,
         max_iter = max_iterations,
         gap_between_prints = gap_between_prints,
         size_of_dataset = size_of_dataset,
         seed = seed,
         fixed_ft = fixed_ft,
         user_choice_subset = user_choice_subset)




