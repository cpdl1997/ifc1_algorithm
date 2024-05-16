import pandas as pd
import numpy as np
import math
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from dataset_gen import dataset_generator as dg
import joblib

# ##################################################################################################################################################################
# #model
# def model_obj(layer_size, lr, max_it=100):
#     clf = MLPClassifier(hidden_layer_sizes=layer_size,
#                     random_state=5,
#                     max_iter=max_it,
#                     verbose=True,
#                     learning_rate_init=lr)
#     return clf


# ##################################################################################################################################################################
# #utility functions
# def sigmoid(x): 
#     return 1 / (1 + math.exp(-x))


# def create_test_train(dataset, test_size):
#     data=pd.read_csv(dataset)
#     data = normalize(data)
#     X=data.drop('Y', axis=1)
#     Y=data['Y']
#     Y[Y>=0.5] = 1
#     Y[Y<0.5] = 0

#     x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=None)
#     return x_train, x_test, y_train, y_test



# def normalize(dataset):
#     for i in dataset.columns:
#         dataset[i] = (dataset[i]-dataset[i].min())/(dataset[i].max()-dataset[i].min())
#     return dataset


# ##################################################################################################################################################################
# #main function
# def implement_model(test_size = 0.3, learning_rate = 0.001, layer_size=(6,5), lower=-20, upper= 20, epoch=100):
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
#     model = model_obj(layer_size, learning_rate, epoch)
#     model.fit(x_train,y_train)

#     # Calcuate test accuracy
#     print("Checking for test data... ")
#     ypred = model.predict(x_test)
#     score = accuracy_score(y_test,ypred)
#     print("Test Accuracy: ", score)

#     #choose to save model
#     while(1):
#         try:
#             ch2 = input("Do you want to save the model?\nEnter your choice[Y\\N]: ").upper()
#             if(ch2=='Y'):
#                 if(ch1==1):
#                     model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "mlp_classifier_model_D1.pkl")
#                 elif(ch1==2):
#                     model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "mlp_classifier_model_D2.pkl")
#                 else:
#                     model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "mlp_classifier_model_D3.pkl")
#                 joblib.dump(model, model_save_path)
#             else:
#                 pass
#             break
#         except IOError:
#             print("Wrong input. Please try again.")


# test_size = 0.3
# learning_rate = 0.01
# layer_size = (6,5)
# lower=-20
# upper= 20
# epochs=200

# # implement_model(test_size, learning_rate, layer_size, lower, upper, epochs)


