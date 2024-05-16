from .baseCF import BaseCF
import torch
import numpy as np
torch.set_printoptions(precision=8)
import torch.nn as nn
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "../")) #for causal-learn, dowhy and graphviz
import scm.graphviz as graphviz
from scm.dowhy import CausalModel
from scm.causallearn.search.FCMBased import lingam

import datetime
import logging
logger = logging.getLogger()
# logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)

#Use to disable logging from other third party libraries - including causallearn, dowhy and graphviz
for _ in logging.root.manager.loggerDict:
    logging.getLogger(_).setLevel(logging.CRITICAL)
    # logging.getLogger(_).disabled = True  # or use this instead of CRITICAL if you'd rather completely disable it

class FourthMethod(BaseCF):

    def __init__(self, data, model, model_scm, labels, graph_dot):
        super().__init__(data, model)
        self.SCM = model_scm
        self.labels = labels
        self.graph_dot = graph_dot
        
    def generate_counterfactuals(self, x_input, features_to_vary, N, learning_rate=0.01, max_iter=100, _lambda=0.01, target =1, gap_between_prints=80):
        '''
        Given: 
            Input: x (1D numpy array)
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
        x = torch.tensor(x_input, dtype=torch.float32, requires_grad = False)
        x_cf = torch.tensor(x_input, dtype=torch.float32, requires_grad = True)
        optimizer = torch.optim.SGD([x_cf], learning_rate)
        target = torch.as_tensor([float(target)])
        target.requires_grad = False
        print_iter=0 #this variable is to ensure that we print only after "gap_between_prints" number pf epochs
        
        # logger.debug("{}: Features_to_vary = {}".format(datetime.datetime.now(), features_to_vary))

        #Just checking if all SCMs generated are same

        # from scm.causallearn.search.FCMBased.lingam.utils import make_dot
        # rand = np.random.randint(low = 1, high = 20000)
        # scm_test = 'SCM{}.pdf'.format(rand)
        # scm_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../logs/scm_dump", scm_test)
        # make_dot(self.SCM.adjacency_matrix_, labels=self.labels).render(str(scm_test), view=False, outfile=scm_path)

        final_classification_loss = 0

        for i in range(max_iter):
            if(print_iter==0 or print_iter%gap_between_prints==0):
                logger.debug("{}: -------------------------------------------------------Iteration {}--------------------------------------------------------".format(datetime.datetime.now(),i+1))
            
            optimizer.zero_grad() #ALTERNATE: x_cf.grad.zero_() which empty the gradients otherwise they keep adding up

            loss1 = torch.sum((x_cf - x)**2)

            y_hat = self.model(x_cf)
            
            if y_hat.data==0:
                y_hat.data = y_hat.data+1e-9

            # if(print_iter==0 or print_iter%gap_between_prints==0):
            #     logger.debug("{}: y_hat = {}".format(datetime.datetime.now(), y_hat))

            bce = nn.BCELoss()
            loss2 = bce(y_hat, torch.as_tensor([float(target)]))

            loss = _lambda*loss1 +  loss2

            if i==max_iter-1:
                final_classification_loss = loss2.item()

            #Control frequency of log statements
            if(print_iter==0 or print_iter%gap_between_prints==0):
                # logger.info("{}: L2 loss (loss1) = {} and Classification loss (loss2) = {} and final computed loss = {}".format(datetime.datetime.now(), torch.round(loss1.data, decimals=3), torch.round(loss2.data, decimals=3), torch.round(loss.data, decimals=3)))
                logger.info("{}: L2 loss (loss1) = {} and Classification loss (loss2) = {} and final computed loss = {}".format(datetime.datetime.now(), loss1.item(), loss2.item(), loss.item()))

            loss.backward() #calculate the gradients d(loss)/d(x_cf) -> Since loss is a scalar value hence this is not an issue, otherwise we need to pass vector to backward()
            x_cf.grad = x_cf.grad * mask

            # if(print_iter==0 or print_iter%gap_between_prints==0):
            #     logger.debug("{}: Gradients: x_cf = {}".format(datetime.datetime.now(), x_cf.grad))

            optimizer.step() #adds the gradient to all the variables


            # if(print_iter==0 or print_iter%gap_between_prints==0):
            #     logger.debug("{}: x_cf before causal change = {}".format(datetime.datetime.now(), x_cf.tolist()))

            #update the causal graph
            x_cf.data = self.update_SCM(x=x, x_cf=x_cf, N = N, graph_dot=self.graph_dot, intervene_list = features_to_vary, print_iter=print_iter, gap_between_prints=gap_between_prints)


            # if(print_iter==0 or print_iter%gap_between_prints==0):
            #     logger.debug("{}: x_cf after causal change = {}".format(datetime.datetime.now(), x_cf.tolist()))

            if(print_iter==0 or print_iter%gap_between_prints==0):
                logger.debug("{}: x_cf generated = {} with \ny_hat = {}".format(datetime.datetime.now(), x_cf.tolist(), y_hat.tolist()))
            
            if(print_iter==0 or print_iter%gap_between_prints==0):
                print_iter=0

            print_iter+=1
        
        # cost = torch.abs(x_cf-x).tolist()[features_to_vary[0]-1] #we only consider L1 distance of the feature being changed
        cost = final_classification_loss #We consider the overall loss

        return x_cf.detach().numpy(), cost


    #Update the causal graph as per the updated counterfactual
    def update_SCM(self, x, x_cf, N, graph_dot, intervene_list=None, print_iter=0, gap_between_prints=100):
        # children = [[] for _ in range(N-1)]
        # for i, ft in enumerate(intervene_list):
        #     children_of_curr = []
        #     outgoing_nodes = self.scm.adjacency_matrix_
        #     for j, val in enumerate(outgoing_nodes):
        #         if val: #if there is an outgoing node from feature ft
        #             children_of_curr.append(j) #keep track of indices to which outgoing edges go to
        #     children[i].append(children_of_curr)

        #     for j in children_of_curr:
        #         # Define Causal Model
        #         model=CausalModel(
        #                 data = self.data,
        #                 treatment=self.data.columns[i],
        #                 outcome='weight',
        #                 graph=self.str_to_dot(graph_dot.source))
        
        # if(print_iter==0 or print_iter%gap_between_prints==0):
        #     logger.debug("{}: Updating SCM for intervention list {}".format(datetime.datetime.now(), intervene_list))

        for i, ft in enumerate(intervene_list):
            list_of_other_ft = list(set([t for t in range(1,N)]) - set([ft])) #The other features except the one which is being changed
            for j, other_ft in enumerate(list_of_other_ft):

                # if(print_iter==0 or print_iter%gap_between_prints==0):
                #     logger.debug("{}: Checking causal effect of {} on {}".format(datetime.datetime.now(), self.data.columns[ft-1], self.data.columns[other_ft-1]))

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
                                                control_value=x[ft-1].item(), #We are studying the effects of change of feature ft value down the stream on feature other_ft
                                                treatment_value=x_cf[ft-1].item(), #Control = original value of feature ft, treatment value is the changed value
                                                confidence_intervals=False,
                                                test_significance=True)
                
                if(print_iter==0 or print_iter%gap_between_prints==0):
                    logger.debug("{}: Causal effect of {} on {} is {}".format(datetime.datetime.now(), self.data.columns[ft-1], self.data.columns[other_ft-1], estimate.value))

                # if(print_iter==0 or print_iter%gap_between_prints==0):
                #     logger.debug("{}: Prior to causal change for current effect x_cf = {}".format(datetime.datetime.now(), x_cf))
                                 
                #estimate.value AKA the causal estimate is the amount of change that is occuring at feature other_ft due to feature ft
                x_cf.data[other_ft-1] = x_cf.data[other_ft-1] + estimate.value #update the causal effect of that variable

                # if(print_iter==0 or print_iter%gap_between_prints==0):
                #     logger.debug("{}: After causal change for current effect x_cf = {}".format(datetime.datetime.now(), x_cf))

            # if(print_iter==0 or print_iter%gap_between_prints==0):
            #     logger.debug("{}: Current x_cf after intervention on {} = {}".format(datetime.datetime.now(), self.data.columns[ft-1], x_cf))
        
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

    #Encode and create the mask vector
    def encoding(self, features_to_vary, N):
        matrix = [0]*(N-1)
        for i in features_to_vary:
            matrix[i-1] = 1
        return np.array(matrix)
