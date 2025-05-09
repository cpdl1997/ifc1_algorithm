from .baseCF import BaseCF
import torch
import numpy as np
import random
torch.set_printoptions(precision=8)
import torch.nn as nn
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")) #for causal-learn, dowhy and graphviz
import scm.graphviz as graphviz
from scm.dowhy import CausalModel
from scm.causallearn.search.FCMBased import lingam

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../")) #for causal-learn, dowhy and graphviz
# from Carla.carla import MLModel

import datetime
import logging
logger = logging.getLogger()
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

#Use to disable logging from other third party libraries - including causallearn, dowhy and graphviz
for _ in logging.root.manager.loggerDict:
    logging.getLogger(_).setLevel(logging.CRITICAL)
    # logging.getLogger(_).disabled = True  # or use this instead of CRITICAL if you'd rather completely disable it

class NinthMethod(BaseCF):

    def __init__(self, data, model, model_scm, labels, graph_dot, history_of_causal_effect):
        super().__init__(data, model)
        self.SCM = model_scm
        self.labels = labels
        self.graph_dot = graph_dot
        self.history_of_causal_effect = history_of_causal_effect
       
    def generate_counterfactuals(self, x_dp, x_input, features_to_vary, N, learning_rate=0.01, max_iter=100, _lambda=0.01, target =1, gap_between_prints=80):
        '''
        Given:
            Original data point: x_dp (1D numpy array)
            Input to function: x_input (1D numpy array)
            Index array of features that can be modified with ordering of change: arr[0] -> arr[1] -> arr[2]....: features_to_vary (list)
            Number of features including output: N (int)
            Model: model
            Learning Rate: learning_rate (float)
            Maximum number of Iterations: max_iter (int)
            Lambda for regularising with L2 distance: _lamba (float)
        Output:
            Counterfactual using all the features in features_to_vary
        '''
        mask = torch.tensor(self.encoding(features_to_vary, N))
        reverse_mask = torch.ones(N-1) - mask
        x = torch.tensor(x_input, dtype=torch.float32, requires_grad = False)
        x_original = torch.tensor(x_dp, dtype=torch.float32, requires_grad = False)
        x_cf = torch.tensor(x_input, dtype=torch.float32, requires_grad = True)
        optimizer = torch.optim.SGD([x_cf], learning_rate)
        target = torch.as_tensor([float(target)])
        target.requires_grad = False
        print_iter=0 #this variable is to ensure that we print only after "gap_between_prints" number pf epochs
        prev_loss = float('inf')
        prev_loss2 = float('inf')

        final_classification_loss = 0

        c1 = torch.empty(N-1)
        c2 = torch.empty(N-1)

        perturbation_val = 100

        i = 0

        while i < max_iter: #We use this instead of range due to us needing to update i later on if needed
       
            if(print_iter==0 or print_iter%gap_between_prints==0):
                logger.debug("{}: -------------------------------------------------------Iteration {}--------------------------------------------------------".format(datetime.datetime.now(),i+1))
           
            optimizer.zero_grad() #ALTERNATE: x_cf.grad.zero_() which empty the gradients otherwise they keep adding up


            with torch.no_grad():
                z1 = (x_cf - x_original)**2
                z2 = z1*reverse_mask
                c1 = torch.sum(z2) #Sum all other elements towards L1 loss, but treat it like constant so their gradient does flow towards these elements not in features_to_vary
                c2 = x_cf*reverse_mask

            loss1 = c1

            for j in range(N-1):
                if mask[j] == 1:
                    loss1 = loss1 + (x_cf[j] - x_original[j])**2
                    c2[j] = x_cf[j]

            y_hat = self.model(c2)

            y_hat_iter=0
            y_curr = y_hat.detach().clone()

            while y_curr.data==0 and y_hat_iter<max_iter:#This counterfactual is doomed -> it gives zero gradient. So we introduce random perturbations to c2[features_to_vary] - perturbations are between [-100, 100).
                with torch.no_grad():
                    for j in features_to_vary:
                        x_cf[j-1] = x_cf[j-1] + random.uniform(0, perturbation_val)
               
                y_curr = self.model(x_cf)
                
                if y_curr.data>0: #We got a good perturbation. Reset since perturbation successfully works
                    #Repeat first all steps so far
                    with torch.no_grad():
                        z1 = (x_cf - x_original)**2
                        z2 = z1*reverse_mask
                        c1 = torch.sum(z2) #Sum all other elements towards L1 loss, but treat it like constant so their gradient does flow towards these elements not in features_to_vary
                        c2 = x_cf*reverse_mask
                    loss1 = c1
                    for j in range(N-1):
                        if mask[j] == 1:
                            loss1 = loss1 + (x_cf[j] - x_original[j])**2
                            c2[j] = x_cf[j]
                    y_hat = self.model(c2)
                elif y_curr.data==0: #We did not get a good perturbation. We now increase the range of perturbation values
                    if y_hat_iter==max_iter-1: #We did not get a permutation in required number of steps
                        logger.info("{}: Due to running out of iterations, counterfactual generation failed.".format(datetime.datetime.now()))
                        return None, -1, None
                    pass
                else: #We failed spectacularly - We now get NaN values for y_hat
                    logger.info("{}: Due to NaN, counterfactual generation failed.".format(datetime.datetime.now()))
                    # logger.debug("{}: torch.isnan(y_curr).data is True {}".format(datetime.datetime.now(), torch.isnan(y_curr).data[0] == True))
                    if torch.isnan(y_curr).data[0] == True: #we have reached NaN value for y_hat. ABORT Immediately
                        return None, -1, None
            
                #Keep trying again
                y_hat_iter +=1
           
            bce = nn.BCELoss()
            loss2 = bce(y_hat, torch.as_tensor([float(target)]))
            loss = _lambda*loss1 +  loss2

            if i==max_iter-1:
                final_classification_loss = loss2.item()

            #Control frequency of log statements
            if(print_iter==0 or print_iter%gap_between_prints==0):
                logger.info("{}: L2 loss (loss1) = {} and Classification loss (loss2) = {} and final computed loss = {} for feature {}.".format(datetime.datetime.now(), loss1.item(), loss2.item(), loss.item(), features_to_vary[0]))


            c2.retain_grad()
            loss1.retain_grad()
            loss2.retain_grad()
            loss.retain_grad()


            loss.backward() #calculate the gradients d(loss)/d(x_cf) -> Since loss is a scalar value hence this is not an issue, otherwise we need to pass vector to backward()

            #check if gradient is too low - in that case, modify the gradients manually to force a signficiant change in the direction of previous gradient
            with torch.no_grad():
                for j in features_to_vary:
                    if abs(x_cf.grad[j-1])<1e-2 and loss2.data>9.9: #We have small gradients and large error: 9.9 is the error when y_hat = 0.00005 -> So we go with 9.9. 1e-2 is arbritarily chosen
                        if x_cf.grad[j-1]<0:
                            x_cf.grad[j-1] = -random.uniform(0, loss2.data)
                        else:
                            x_cf.grad[j-1] = random.uniform(0, loss2.data)


            #check if loss increases per iteration and classification loss doesn't change (This algorithm shows a tendency to reduce classification loss to zero in the first try) - if it does, flip the sign of gradient to force a direction change
            if prev_loss<loss.item() and prev_loss2>loss2.item(): #if no change between iterations, even then force a change
                x_cf.grad.data = x_cf.grad.data * (-1)
           
            prev_loss2 = loss2.item()
            prev_loss = loss.item()

            optimizer.step() #adds the gradient to all the variables

            #update the causal graph
            x_cf.data = self.update_SCM(x_cf_prev=x, x_cf=x_cf, N = N, graph_dot=self.graph_dot, intervene_list = features_to_vary, print_iter=print_iter, gap_between_prints=gap_between_prints)

            if(print_iter==0 or print_iter%gap_between_prints==0):
                print_iter=0

            with torch.no_grad():
                x = x_cf.detach().clone()
                x.requires_grad=False

            print_iter+=1
            i+=1
            logger.debug("{}: ------------------------------------------------------- End of Iteration {}-------------------------------------------------------".format(datetime.datetime.now(), i+1))
       
        cost = final_classification_loss #We consider the overall loss

        return x_cf.detach().numpy(), cost, y_hat.data


    #Update the causal graph as per the updated counterfactual
    '''
    Some of the benefits of using LinGAM is that:
    1) It always provides a DAG
    2) The causal effect between f1 and f2 is linear, so if f1 takes two values x1 (control) and x2 (treatment) and resulting f2 is y
    then if f1 takes two values x3 (control) and x4 (treatment), resulting f2 is y*(x4-x3)/(|x2-x1|) -> This allows us to optimize by storing previous runs and using that
    to calculate newer values. => This fact has been checked via testing
    '''
    def update_SCM(self, x_cf_prev, x_cf, N, graph_dot, intervene_list=None, print_iter=0, gap_between_prints=100):
        for i, ft in enumerate(intervene_list):
            list_of_other_ft = list(set([t for t in range(1,N)]) - set([ft])) #The other features except the one which is being changed
            for j, other_ft in enumerate(list_of_other_ft):  
                est = 0 #Keeps track of causal estimate
                ctrl_val = x_cf_prev[ft-1].item() #Value of control = Previous value of x_cf[ft-1] before gradient applied
                trtment_val = x_cf[ft-1].item() #Value of treatment - Current value of x_cf[ft-1] after gradient applied
                if ctrl_val!=trtment_val:
                    if (ft, other_ft) in self.history_of_causal_effect: #We have already explored this relationship
                       
                        if(print_iter==0 or print_iter%gap_between_prints==0):
                            logger.debug("{}: This relationship {}->{} has been explored already! Value for this scenario is based on original control = {} and treatment = {} on {}, and original value is {}. Current control = {} and treatment = {}.".format(datetime.datetime.now(),
                                                                                                                                                                                    self.data.columns[ft-1],
                                                                                                                                                                                    self.data.columns[other_ft-1],
                                                                                                                                                                                    self.history_of_causal_effect[(ft, other_ft)][0],
                                                                                                                                                                                    self.history_of_causal_effect[(ft, other_ft)][1],
                                                                                                                                                                                    self.data.columns[ft-1],
                                                                                                                                                                                    self.history_of_causal_effect[(ft, other_ft)][2],
                                                                                                                                                                                    ctrl_val,
                                                                                                                                                                                    trtment_val)
                                                                                                                                                                                    )

                        est = self.history_of_causal_effect[(ft, other_ft)][2] * (trtment_val - ctrl_val)/(abs(self.history_of_causal_effect[(ft, other_ft)][1] - self.history_of_causal_effect[(ft, other_ft)][0]))

                    else: #This is a new causal relationship
                        model=CausalModel(
                                    data = self.data,
                                    treatment=self.data.columns[ft-1],
                                    outcome=self.data.columns[other_ft-1],
                                    graph=self.str_to_dot(graph_dot.source))
                        # Identification
                        identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
                        # Estimation
                        estimate = model.estimate_effect(identified_estimand,
                                                        method_name="backdoor.linear_regression",
                                                        control_value=ctrl_val, #We are studying the effects of change of feature ft value down the stream on feature other_ft
                                                        treatment_value=trtment_val, #Control = original value of feature ft, treatment value is the changed value
                                                        confidence_intervals=False,
                                                        test_significance=True)

                        est = estimate.value

                        if(print_iter==0 or print_iter%gap_between_prints==0):
                            logger.debug("{}: NEW CAUSAL INFERENCE: {}->{} with control Value {} and treatment value {}.".format(datetime.datetime.now(), self.data.columns[ft-1], self.data.columns[other_ft-1], ctrl_val, trtment_val))

                        #Add it to history so far -> (Original control value, Original Treatment Value, Causal Effect)
                        self.history_of_causal_effect[(ft, other_ft)] = (ctrl_val, trtment_val, est)    
                   

                    if(print_iter==0 or print_iter%gap_between_prints==0):
                        logger.debug("{}: Causal effect of {} on {} is {}".format(datetime.datetime.now(), self.data.columns[ft-1], self.data.columns[other_ft-1], est))

                    #estimate.value AKA the causal estimate is the amount of change that is occuring at feature other_ft due to feature ft
                    x_cf.data[other_ft-1] = x_cf.data[other_ft-1] + est #update the causal effect of that variable

        return x_cf


    def make_graph(self, adjacency_matrix, labels=None):
        idx = np.abs(adjacency_matrix) > 0.01
        dirs = np.where(idx)
        d = graphviz.Digraph(engine='dot')
        names = labels if labels else [f'x{i}' for i in range(len(adjacency_matrix))]
        for name in names:
            d.node(name)
        for to, from_, coef in zip(dirs[0], dirs[1], adjacency_matrix[idx]):
            d.edge(names[from_], names[to], label=str(coef))
        return d

    def str_to_dot(self, string):
        '''
        Converts input string from graphviz library to valid DOT graph format.
        '''
        graph = string.strip().replace('\n', ';').replace('\t','')
        graph = graph[:9] + graph[10:-2] + graph[-1] # Removing unnecessary characters from string
        return graph

    #Encode and create the mask vector: Returns np.array
    def encoding(self, features_to_vary, N):
        matrix = [0]*(N-1)
        for i in features_to_vary:
            matrix[i-1] = 1
        return np.array(matrix)