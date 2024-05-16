import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# from keras.callbacks import ModelCheckpoint
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from dataset_gen import dataset_generator as dg


import logging
import time
log_file_name = "Log_custom_sequential.log"
log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../logs", log_file_name)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(log_path, mode='w')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

import warnings
warnings.filterwarnings('ignore')

# ##################################################################################################################################################################
# #model
# def model_obj(train, layer_size, lr):
#     model = nn.Sequential(
#         nn.Linear(in_features = train.shape[1], out_features = 256),
#         nn.ReLu(),
#         nn.Linear(in_features = 256, out_features = 256),
#         nn.ReLu(),
#         nn.Linear(in_features = 256, out_features = 256),
#         nn.ReLu(),
#         nn.Linear(in_features = 256, out_features = 256),
#         nn.ReLu(),
#         nn.Linear(in_features = 256, out_features = 1),
#         nn.ReLu()
#     )
#     return model


# # def model_obj(train, layer_size, lr):
# #     NN_model = Sequential()
# #     # The Input Layer :
# #     NN_model.add(Dense(256, kernel_initializer='normal',input_dim = train.shape[1], activation='relu'))

# #     # The Hidden Layers :
# #     NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
# #     NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
# #     NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

# #     # The Output Layer :
# #     NN_model.add(Dense(1))

# #     # Compile the network :
# #     NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
# #     NN_model.summary()
# #     return NN_model


# ##################################################################################################################################################################
# #utility functions
# def create_test_train(dataset, test_size):
#     data=pd.read_csv(dataset)
#     data = normalize(data)
#     X=data.drop('Y', axis=1)
#     Y=data['Y']
#     x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=0)
#     return x_train, x_test, y_train, y_test



# def normalize(dataset):
#     for i in dataset.columns:
#         dataset[i] = (dataset[i]-dataset[i].min())/(dataset[i].max()-dataset[i].min())
#     return dataset


# ##################################################################################################################################################################
# #main function
# def implement_model(test_size = 0.3, learning_rate = 0.001, hidden_layer_size = (6,5), lower=-20, upper= 20, epoch=500):
#     #paramaters

#     #generate datasets
#     while(1):
#         try:
#             ch = input("\nDo you want to generate the dataset?\nEnter your choice[Y\\N]: ").upper()
#             if(ch=='Y'):
#                 dg.generateDatapoints(lower, upper)
#             else:
#                 pass
#             break
#         except IOError:
#             print("Wrong input. Please try again.")

#     #choose datasets
#     while(1):
#         try:
#             ch1 = input("Choose the dataset you want:\n1) dataset1.csv\n2) dataset2.csv\n3) dataset3.csv\nEnter your choice: ")
#             if(ch1!='1' and ch1!='2' and ch1!='3'):
#                 print("Incorrect Input.")
#             else:
#                 ch1=int(ch1)
#                 if(ch1==1):
#                     dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "dataset1.csv")
#                 elif(ch1==2):
#                     dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "dataset2.csv")
#                 else:
#                     dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "dataset3.csv")
#             break
#         except IOError:
#             print("Wrong input. Please try again.")

#     #training
#     x_train, x_test, y_train, y_test = create_test_train(dataset, test_size)
#     model = model_obj(x_train, hidden_layer_size, learning_rate)
#     model.fit(x_train,y_train, epochs=epoch)

#     # Calcuate test accuracy
#     print("Checking for test data... ")
#     score = model.evaluate(x_test,y_test)
#     print("Test Accuracy: ", score)

#     #choose to save model
#     while(1):
#         try:
#             ch2 = input("Do you want to save the model?\nEnter your choice[Y\\N]: ").upper()
#             if(ch2=='Y'):
#                 if(ch1==1):
#                     model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential_model_D1.pkl")
#                 elif(ch1==2):
#                     model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential_model_D2.pkl")
#                 else:
#                     model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential_model_D3.pkl")
#                 joblib.dump(model, model_save_path)
#             else:
#                 pass
#             break
#         except IOError:
#             print("Wrong input. Please try again.")
    


# test_size = 0.3
# learning_rate = 0.001
# hidden_layer_size = (6,5)
# lower=-20
# upper= 20
# epochs=10

# # implement_model(test_size, learning_rate, hidden_layer_size, lower, upper, epochs)


#         nn.Linear(in_features = train.shape[1], out_features = 256),
#         nn.ReLu(),
#         nn.Linear(in_features = 256, out_features = 256),
#         nn.ReLu(),
#         nn.Linear(in_features = 256, out_features = 256),
#         nn.ReLu(),
#         nn.Linear(in_features = 256, out_features = 256),
#         nn.ReLu(),
#         nn.Linear(in_features = 256, out_features = 1),
#         nn.ReLu()


class Custom_Model(nn.Module):
    def __init__(self, in_features):
        super(Custom_Model, self).__init__()
        self.lin1 = nn.Linear(in_features = in_features, out_features = 10)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(in_features = 10, out_features = 10)
        self.act2 = nn.ReLU()
        self.lin3 = nn.Linear(in_features = 10, out_features = 5)
        self.act3 = nn.ReLU()
        self.lin4 = nn.Linear(in_features = 5, out_features = 5)
        self.act4 = nn.ReLU()
        self.lin5 = nn.Linear(in_features = 5, out_features = 1)
        self.act5 = nn.Sigmoid()
        
    def forward(self, train):
        x = self.act1(self.lin1(train))
        x = self.act2(self.lin2(x))
        x = self.act3(self.lin3(x))
        x = self.act4(self.lin4(x))
        x = self.act5(self.lin5(x))
        return x
    


class Custom_Model2(nn.Module):
    def __init__(self, in_features):
        super(Custom_Model2, self).__init__()
        self.lin1 = nn.Linear(in_features = in_features, out_features = 1)
        
    def forward(self, train):
        x = F.sigmoid(self.lin1(train))
        return x
    


class Custom_Model3(nn.Module):
    def __init__(self, in_features):
        super(Custom_Model3, self).__init__()
        self.layer1 = nn.Linear(in_features= in_features, out_features=60)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(60, 60)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(60, 60)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(60, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, train):
        x = self.act1(self.layer1(train))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x