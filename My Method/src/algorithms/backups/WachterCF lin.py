# sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
# import algorithms 
from .baseCF import BaseCF #use "from .baseCF import BaseCF" for Ubuntu
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import copy
from dataset_gen import dataset_generator as dg

import datetime
import logging
import time
logger = logging.getLogger()

class WachterCF(BaseCF):

    def __init__(self, data, model):
        super().__init__(data, model)

    # def generate_counterfactuals(self, x, features_to_vary, N, learning_rate=0.01, max_iter=100, _lambda=0.01, target =1):
    #     '''
    #     Given: 
    #         Input: x (1D numpy array)
    #         Index array of features that can be modified: features_to_vary (list)
    #         Number of features including output: N (int)
    #         Model: model
    #         Learning Rate: learning_rate (float)
    #         Maximum number of Iterations: max_iter (int)
    #         Lambda for regularising with L2 distance: _lamba (float)
    #     Output:
    #         Counterfactual using all the features in features_to_vary
    #     '''
    #     mask = torch.Tensor(self.encoding(features_to_vary, N))
    #     x = torch.Tensor(x)
    #     # x_cf = torch.rand(x.shape)*torch.max(x)
    #     x_cf_changeable = copy.deepcopy(x)
    #     x_cf_nonchangeable = copy.deepcopy(x)
    #     x_cf_changeable.requires_grad_(True)
    #     # x_cf_nonchangeable.requires_grad=False
    #     logger.debug("{}: x.is_leaf = {}".format(datetime.datetime.now(), x.is_leaf))
    #     logger.debug("{}: x_cf_changeable.is_leaf 1= {}".format(datetime.datetime.now(), x_cf_changeable.is_leaf))
    #     logger.debug("{}: x_cf_nonchangeable.is_leaf 1= {}".format(datetime.datetime.now(), x_cf_nonchangeable.is_leaf))

    #     optimizer = torch.optim.Adam([x_cf_changeable], learning_rate)
    #     logger.debug("{}: x = {}".format(datetime.datetime.now(), x))
    #     logger.debug("{}: ft_encoded = {}".format(datetime.datetime.now(), mask))
    #     logger.debug("{}: x_cf_changeable.is_leaf 2= {}".format(datetime.datetime.now(), x_cf_changeable.is_leaf))

    #     for i in range(max_iter):
    #         logger.debug("{}: x_cf_changeable 3= {}".format(datetime.datetime.now(), x_cf_changeable))
    #         logger.debug("{}: x_cf_nonchangeable 4= {}".format(datetime.datetime.now(), x_cf_nonchangeable))
    #         optimizer.zero_grad() #ALTERNATE: x_cf.grad.zero_() which empty the gradients otherwise they keep adding up
    #         # x_cf_changeable = mask*x_cf_changeable + (1-mask)*x_cf_nonchangeable
    #         # x_cf_changeable.requires_grad = True
    #         logger.debug("{}: x_cf_changeable.is_leaf 5= {}".format(datetime.datetime.now(), x_cf_changeable.is_leaf))
    #         loss = self.compute_loss(x, x_cf_changeable, self.model, _lambda, target) #find loss using x and x_cf
    #         print(loss)
    #         # if(loss==0):
    #         #     break
    #         # loss = torch.Tensor(loss) #find loss using x and x_cf
    #         logger.debug("{}: Gradients = {}".format(datetime.datetime.now(), x_cf_changeable.grad))
    #         x_cf_changeable.retain_grad()
    #         loss.backward(retain_graph=True) #calculate the gradients d(loss)/d(x_cf) -> Since loss is a scalar value hence this is not an issue, otherwise we need to pass vector to backward()
    #         optimizer.step() #adds the gradient to all the variables
            

    #     return x_cf_changeable
        
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
        # mask.requires_grad = False
        x = torch.tensor(x_input, dtype=torch.float32, requires_grad = False)
        
        lin = nn.Linear(N, 1)
        lin.weight.data = x.data
        optimizer = torch.optim.Adam([lin.weight], learning_rate)
        print_iter=1 #this variable is to ensure that we print only after "gap_between_prints" number pf epochs

        

        for i in range(max_iter):
            optimizer.zero_grad() #ALTERNATE: x_cf.grad.zero_() which empty the gradients otherwise they keep adding up

            lin.weight.data = lin.weight.data*mask.data + (1-mask.data)*x.data
            #Use whatever distance you want
            # loss1 = torch.sum(torch.pow(torch.subtract(x, x_cf),2)) #L2 distance loss
            mse = nn.MSELoss()

            if(print_iter%gap_between_prints==0):
                logger.debug("********************************************************Iteration {}********************************************************".format(i+1))
                logger.debug("{}: Lin.weight = {}".format(datetime.datetime.now(), lin.weight.tolist()))
            loss1 = mse(x, lin.weight)
            # mse = nn.L1Loss()
            # loss1 = mse(x_cf, x) #L1 distance loss
            # loss1.retain_grad()
            # loss1.register_hook(lambda grad: logger.debug("{}: loss1.grad = {}".format(datetime.datetime.now(), grad))) #print the gradient reaching loss1
            

            bce = nn.BCELoss()
            # y_hat = self.find_value(x_cf, self.model)
            y_hat = self.model(lin.weight)
            # y_hat.retain_grad()
            # y_hat.register_hook(lambda grad: logger.debug("{}: y_hat.grad = {}".format(datetime.datetime.now(), grad))) #print the gradient reaching y_hat
            

            loss2 = bce(y_hat, torch.as_tensor([float(target)])) #Classification loss
            # loss2.retain_grad()
            # loss2.register_hook(lambda grad: logger.debug("{}: loss2.grad = {}".format(datetime.datetime.now(), grad))) #print the gradient reaching loss2
            

            loss = loss1 + _lambda * loss2
            # loss = _lambda * loss2
            # loss.retain_grad()
            # loss.register_hook(lambda grad: logger.debug("{}: loss.grad = {}".format(datetime.datetime.now(), grad))) #print the gradient reaching loss
            

            #Control frequency of log statements
            if(print_iter%gap_between_prints==0):
                # logger.debug("{}: Tree Check: x.is_leaf= {}, x_cf.is_leaf= {}, mask.is_leaf= {}".format(datetime.datetime.now(), x.is_leaf, x_cf.is_leaf, mask.is_leaf))
                logger.debug("{}: L2 loss (loss1) = {} and Classification loss (loss2) = {} and final computed loss = {}".format(datetime.datetime.now(), loss1, loss2, loss))
                logger.debug("{}: lin.weight = {} with y_hat = {} with original x = {}".format(datetime.datetime.now(), lin.weight.tolist(), y_hat.tolist(), x.tolist()))
                # logger.debug("{}: Tree-> See Below: ".format(datetime.datetime.now()))
                # self.print_graph(loss.grad_fn, 0) #print the computation graph

            loss.backward() #calculate the gradients d(loss)/d(x_cf) -> Since loss is a scalar value hence this is not an issue, otherwise we need to pass vector to backward()
            # x.grad = None
            # x_cf.grad = x_cf.grad*10
            # x_cf.grad = x_cf.grad * mask
            optimizer.step() #adds the gradient to all the variables

            if(print_iter%gap_between_prints==0):
                logger.debug("{}: Gradients: x = {}, lin.weight = {}, mask = {}".format(datetime.datetime.now(), x.grad, lin.weight.grad, mask.grad))
                print_iter=0

            print_iter+=1

        
        return lin.weight


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

    #Return the value of x from the model
    def find_value(self, x, model):
        # x = x.to(torch.float32) #need to convert to float32 in order to run the model -> must match the format of the weights
        # #NOTE FOR FUTURE: THIS "torch.float32" DATATYPE SHOULD BE VIA A PARAMETER, NOT HARDCODED
        # return torch.reshape(model(x), (1,-1))[0]
        return(model(x))


class WachterLoss(nn.Module):
    def __init__(self):
        super(WachterLoss, self).__init__()

    # def calculate_loss(self, x, x_cf, model, _lambda = 0.1, target = 1):
    #     loss1 = torch.sum(torch.pow(torch.subtract(x_cf, x),2)) #L2 distance loss
    #     bce = nn.BCEWithLogitsLoss()
    #     y_hat = self.find_value(x_cf, model)
    #     loss2 = bce(y_hat, torch.as_tensor([float(target)])) #Classification loss
    #     logger.debug("{}: x_cf = {} with y_hat = {}".format(datetime.datetime.now(), x_cf.tolist(), y_hat.tolist()))
    #     print("Loss1: ", loss1)
    #     print("Loss2: ", loss2)
    #     logger.debug("{}: x_cf.grad: = {}".format(datetime.datetime.now(), x_cf.grad))
    #     logger.debug("{}: L2 loss (loss1) = {} and Classification loss (loss2) = {} and final computed loss = {}".format(datetime.datetime.now(), loss1, loss2, loss1 + _lambda * loss2))
    #     return loss1 + _lambda * loss2

    def find_value(self, x, model):
        x = x.to(torch.float32) #need to convert to float32 in order to run the model -> must match the format of the weights
        #NOTE FOR FUTURE: THIS "torch.float32" DATATYPE SHOULD BE VIA A PARAMETER, NOT HARDCODED
        return torch.reshape(model(x), (1,-1))[0]