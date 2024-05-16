# sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
# import algorithms 
from .baseCF import BaseCF
from scm import Node, SCM
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import datetime
import logging
logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

class SecondMethodCF(BaseCF):

    def __init__(self, data, model, scm:SCM.SCM, alpha_weights = None, alpha_sum=1, alpha_multiplier=1.1):
        super().__init__(data, model)
        self.SCM = scm
        self.alpha_weights = alpha_weights
        self.alpha_sum = alpha_sum
        self.alpha_multiplier = alpha_multiplier
        
    def generate_counterfactuals(self, x_input, features_to_vary, N, learning_rate=0.01, max_iter=100, _lambda=0.01, target =1, gap_between_prints=80):
        '''
        Given: 
            Input: x (1D numpy array)
            Index array of features that can be modified (Preference of index i > Preference of index j if i<j): features_to_vary (list)
            Number of features including output: N (int)
            Weights to be given to each term in L1 norm for regularisation: alpha_weights (numpy array)
            Sum of alpha values in alpha_weights: alpha_sum (int) => Case 1 (default): Sums to 1, Case 2: Sums to max (1, loss2), Case 3: Sums to some custom sum
            How alpha(i+1) is related to alpha(i): alpha_multiplier (float) =>  alpha(i+1) = alpha_multiplier * alpha(i)
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

        #create the weights to be multiplied with each
        if self.alpha_weights is None:
            self.alpha_weights = self.generate_alpha_weights(features_to_vary, N-1)

        self.alpha_weights = torch.tensor(self.alpha_weights, dtype=torch.float32, requires_grad = False)
        logger.info("{}: Alpha weights = {}".format(datetime.datetime.now(), self.alpha_weights))
        
        for i in range(max_iter):
            if(print_iter==0 or print_iter%gap_between_prints==0):
                logger.debug("********************************************************Iteration {}********************************************************".format(i+1))
            
            optimizer.zero_grad() #ALTERNATE: x_cf.grad.zero_() which empty the gradients otherwise they keep adding up

            #Use whatever distance you want
            s1 = x_cf - x
            s2 = torch.abs(s1)

            s3 = self.alpha_weights*s2

            loss1 = torch.sum(s2)
            y_hat = self.model(x_cf)

            if y_hat.data==0:
                y_hat.data = y_hat.data+1e-5

            if(print_iter==0 or print_iter%gap_between_prints==0):
                logger.debug("{}: y_hat = {}".format(datetime.datetime.now(), y_hat))


            s4 = torch.log(y_hat)
            loss2 = -s4 #Classification loss
            s5 = _lambda*loss1
            loss = s5 +  loss2

            #Control frequency of log statements
            if(print_iter==0 or print_iter%gap_between_prints==0):
                logger.info("{}: Weighted L1 loss (loss1) = {} and Classification loss (loss2) = {} and final computed loss = {}".format(datetime.datetime.now(), torch.round(loss1.data, decimals=3), torch.round(loss2.data, decimals=3), torch.round(loss.data, decimals=3)))
                # self.print_graph(loss.grad_fn, 0) #print the computation graph

            loss.backward() #calculate the gradients d(loss)/d(x_cf) -> Since loss is a scalar value hence this is not an issue, otherwise we need to pass vector to backward()
            x_cf.grad = x_cf.grad * mask

            if(print_iter==0 or print_iter%gap_between_prints==0):
                logger.debug("{}: Gradients: x_cf = {}".format(datetime.datetime.now(), x_cf.grad))

            optimizer.step() #adds the gradient to all the variables
            
            intervene_ls = []
            for ft in features_to_vary:
                if ft !='':
                    intervene_ls.append(ft-1)

            x_cf.data = self.update_SCM(x_cf=x_cf, intervene_list = intervene_ls , print_iter=print_iter, gap_between_prints=gap_between_prints)

            if(print_iter==0 or print_iter%gap_between_prints==0):
                logger.debug("{}: x_cf generated = {} with \ny_hat = {}".format(datetime.datetime.now(), x_cf.tolist(), y_hat.tolist()))

            if(print_iter==0 or print_iter%gap_between_prints==0):
                print_iter=0

            print_iter+=1

        return x_cf


    def generate_alpha_weights(self, features_to_vary, N):
        ls = np.zeros(N)
        current_alpha = self.alpha_sum*(self.alpha_multiplier - 1)/(self.alpha_multiplier**N - 1)
        # ls[features_to_vary[0]-1] = current_alpha
        for i in range(0, N):
            logger.debug("{}: current_alpha = {} at i = {}".format(datetime.datetime.now(), current_alpha, i))
            if i<len(features_to_vary):
                ls[features_to_vary[i]-1] = current_alpha
                current_alpha = current_alpha*self.alpha_multiplier
        return ls


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
            if i != '':
                matrix[i-1] = 1
        return np.array(matrix)
