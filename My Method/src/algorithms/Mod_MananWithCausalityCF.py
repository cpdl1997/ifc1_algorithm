# sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
# import algorithms 
from .baseCF import BaseCF
from scm import Node, SCM
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import itertools

import datetime
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.DEBUG)

class MananWithCausalityCF(BaseCF):

    def __init__(self, data, model, scm):
        super().__init__(data, model)
        self.SCM = scm
        
    def generate_counterfactuals(self, x_input, features_to_vary, N, learning_rate=0.01, max_iter=100, _lambda=0.01, target =1, gap_between_prints=80):
        '''
        Given: 
            Input: x (1D numpy array)
            Index array of features that can be modified: features_to_vary (list)
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
        
        for i in range(max_iter):
            if(print_iter==0 or print_iter%gap_between_prints==0):
                logger.debug("********************************************************Iteration {}********************************************************".format(i+1))
            
            optimizer.zero_grad() #ALTERNATE: x_cf.grad.zero_() which empty the gradients otherwise they keep adding up

            # s1 = x_cf - x
            # s2 = s1**2
            # loss1 = torch.sum(s2)
            loss1 = torch.sum((x_cf - x)**2)

            y_hat = self.model(x_cf)
            
            if y_hat.data==0:
                y_hat.data = y_hat.data+1e-9

            if(print_iter==0 or print_iter%gap_between_prints==0):
                logger.debug("{}: y_hat = {}".format(datetime.datetime.now(), y_hat))

            # s3 = torch.log(y_hat)
            # loss2 = -s3 #Classification loss
            bce = nn.BCELoss()
            loss2 = bce(y_hat, torch.as_tensor([float(target)]))

            # s4 = _lambda*loss1
            # loss = s4 +  loss2

            loss = _lambda*loss1 +  loss2

            #Control frequency of log statements
            if(print_iter==0 or print_iter%gap_between_prints==0):
                logger.info("{}: L2 loss (loss1) = {} and Classification loss (loss2) = {} and final computed loss = {}".format(datetime.datetime.now(), torch.round(loss1.data, decimals=3), torch.round(loss2.data, decimals=3), torch.round(loss.data, decimals=3)))
                # self.print_graph(loss.grad_fn, 0) #print the computation graph

            loss.backward() #calculate the gradients d(loss)/d(x_cf) -> Since loss is a scalar value hence this is not an issue, otherwise we need to pass vector to backward()
            x_cf.grad = x_cf.grad * mask

            if(print_iter==0 or print_iter%gap_between_prints==0):
                logger.debug("{}: Gradients: x_cf = {}".format(datetime.datetime.now(), x_cf.grad))

            # optimizer.step() #adds the gradient to all the variables 
            '''
            HERE IS THE OPTIMZER.STEP() function -> We are manually doing this in the following code.
            '''
            
            list_of_permutations = [list(x) for x in list(itertools.permutations(features_to_vary))]
            
            for x in d
            # 
            with torch.no_grad():
                new_val = learning_rate*x_cf.grad
                best_order = []
                lowest_distance = float('inf')
                dist_in_order = 0
                for permutation in list_of_permutations: #choose any particular order you would apply the gradients in
                    x_cf_temp = x_cf.deepcopy()
                    new_val_temp = new_val.deepcopy()
                    for i in permutation:
                        new_val_temp[i-1] = x_cf_temp[i-1] + mask[i-1]*new_val_temp[i-1]
                        x_cf_temp.data, dist = self.update_SCM(x_cf=x_cf_temp, intervene_list = [i] , print_iter=print_iter, gap_between_prints=gap_between_prints)
                        dist_in_order += dist
                    
                    if dist_in_order<lowest_distance:
                        lowest_distance = dist_in_order
                        best_order = permutation

                for i in best_order:
                    new_val[i-1] = x_cf[i-1] + mask[i-1]*new_val[i-1]
                    new_val.data, dist = self.update_SCM(x_cf=new_val, intervene_list = [i] , print_iter=print_iter, gap_between_prints=gap_between_prints)  
                
                #needs some fixing

                x_cf.copy_(new_val)


            if(print_iter==0 or print_iter%gap_between_prints==0):
                logger.debug("{}: x_cf generated = {} with \ny_hat = {}".format(datetime.datetime.now(), x_cf.tolist(), y_hat.tolist()))
            
            if(print_iter==0 or print_iter%gap_between_prints==0):
                print_iter=0

            print_iter+=1

        return x_cf


    #Update the causal graph as per the updated counterfactual
    def update_SCM(self, x_cf, intervene_list=None, print_iter=0, gap_between_prints=100):
        return self.SCM.calculate(x_cf, intervene_list, self.SCM.root, print_iter, gap_between_prints)

    #Print the computation graph
    def print_graph(self, g, level=0):
        if g == None: return
        logger.debug('------------->{} at level {}'.format(g, level))
        for subg in g.next_functions:
            self.print_graph(subg[0], level+1)


    #Encode and create the mask vector
    def encoding(self, features_to_vary, N):
        matrix = [0]*(N-1)
        for i in features_to_vary:
            matrix[i-1] = 1
        return np.array(matrix)
