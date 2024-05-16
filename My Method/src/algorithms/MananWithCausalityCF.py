# sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
# import algorithms 
from .baseCF import BaseCF
from scm import Node, SCM
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import random

import datetime
import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
# logger.setLevel(logging.DEBUG)

class MananWithCausalityCF(BaseCF):

    def __init__(self, data, model, scm:SCM.SCM):
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
        cost = 0
        
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

            optimizer.step() #adds the gradient to all the variables

            #update the causal graph
            x_cf.data = self.update_SCM(x_cf=x_cf, intervene_list = [ft-1 for ft in features_to_vary] , print_iter=print_iter, gap_between_prints=gap_between_prints)

            if(print_iter==0 or print_iter%gap_between_prints==0):
                logger.debug("{}: x_cf generated = {} with \ny_hat = {}".format(datetime.datetime.now(), x_cf.tolist(), y_hat.tolist()))
            
            if(print_iter==0 or print_iter%gap_between_prints==0):
                print_iter=0

            print_iter+=1
        
        cost += torch.sum(torch.abs(x_cf-x)).tolist()
        return x_cf, cost


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
    






class MananWithCausalityCF2(BaseCF):

    def __init__(self, data, model, scm:SCM.SCM):
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
        reverse_mask = torch.ones(N-1) - mask
        x = torch.tensor(x_input, dtype=torch.float32, requires_grad = False)
        # x_original = torch.tensor(x_dp, dtype=torch.float32, requires_grad = False)
        x_cf = torch.tensor(x_input, dtype=torch.float32, requires_grad = True)
        optimizer = torch.optim.SGD([x_cf], learning_rate)
        target = torch.as_tensor([float(target)])
        target.requires_grad = False
        print_iter=0 #this variable is to ensure that we print only after "gap_between_prints" number pf epochs
        prev_loss = float('inf')
        prev_loss2 = float('inf')

        # final_classification_loss = 0

        c1 = torch.empty(N-1)
        c2 = torch.empty(N-1)

        perturbation_val = 100

        i = 0
        
        while i < max_iter: #We use this instead of range due to us needing to update i later on if needed
       
            if(print_iter==0 or print_iter%gap_between_prints==0):
                logger.debug("{}: -------------------------------------------------------Iteration {}--------------------------------------------------------".format(datetime.datetime.now(),i+1))
           
            optimizer.zero_grad() #ALTERNATE: x_cf.grad.zero_() which empty the gradients otherwise they keep adding up


            with torch.no_grad():
                z1 = (x_cf - x)**2
                z2 = z1*reverse_mask
                c1 = torch.sum(z2) #Sum all other elements towards L1 loss, but treat it like constant so their gradient does flow towards these elements not in features_to_vary
                c2 = x_cf*reverse_mask

            loss1 = c1

            for j in range(N-1):
                if mask[j] == 1:
                    loss1 = loss1 + (x_cf[j] - x[j])**2
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
                        z1 = (x_cf - x)**2
                        z2 = z1*reverse_mask
                        c1 = torch.sum(z2) #Sum all other elements towards L1 loss, but treat it like constant so their gradient does flow towards these elements not in features_to_vary
                        c2 = x_cf*reverse_mask
                    loss1 = c1
                    for j in range(N-1):
                        if mask[j] == 1:
                            loss1 = loss1 + (x_cf[j] - x[j])**2
                            c2[j] = x_cf[j]
                    y_hat = self.model(c2)
                elif y_curr.data==0: #We did not get a good perturbation. We now increase the range of perturbation values
                    if y_hat_iter==max_iter-1: #We did not get a permutation in required number of steps
                        logger.info("{}: Due to running out of iterations, counterfactual generation failed.".format(datetime.datetime.now()))
                        return None, -1
                    pass
                else: #We failed spectacularly - We now get NaN values for y_hat
                    logger.info("{}: Due to NaN, counterfactual generation failed.".format(datetime.datetime.now()))
                    if torch.isnan(y_curr).data[0] == True: #we have reached NaN value for y_hat. ABORT Immediately
                        return None, -1
            
                #Keep trying again
                y_hat_iter +=1
           
            bce = nn.BCELoss()
            loss2 = bce(y_hat, torch.as_tensor([float(target)]))
            loss = _lambda*loss1 +  loss2

            if i==max_iter-1:
                cost = loss2.item()

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
            if prev_loss<loss.item() and prev_loss2>loss2.item():
                x_cf.grad.data = x_cf.grad.data * (-1)
           
            prev_loss2 = loss2.item()
            prev_loss = loss.item()

            optimizer.step() #adds the gradient to all the variables

            #update the causal graph
            x_cf.data = self.update_SCM(x_cf=x_cf, intervene_list = [ft-1 for ft in features_to_vary] , print_iter=print_iter, gap_between_prints=gap_between_prints)

            if(print_iter==0 or print_iter%gap_between_prints==0):
                print_iter=0

            print_iter+=1
            i+=1
            logger.debug("{}: ------------------------------------------------------- End of Iteration {}-------------------------------------------------------".format(datetime.datetime.now(), i+1))
       
        cost += torch.sum(torch.abs(x_cf-x)).tolist()
        return x_cf, cost


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
