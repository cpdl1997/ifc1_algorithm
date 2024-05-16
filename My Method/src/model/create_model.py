import pandas as pd
import numpy as np
import math
import torch
import torch.nn as nn
from torchvision.ops import MLP
import torch.optim as optim
import custom_sequential as CS
from torch.utils.data import Dataset, TensorDataset
from torchmetrics.classification import BinaryAccuracy
from sklearn.preprocessing import LabelEncoder
# from keras.callbacks import ModelCheckpoint
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
from dataset_gen import dataset_generator as dg
import datetime

import logging
import time
log_file_name = "Log_create_model.log"
log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../logs", log_file_name)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(log_path, mode='w')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

import warnings
warnings.filterwarnings('ignore')

##################################################################################################################################################################
#model
def model_maker(model_type, train, lr = 0.01, layer_size = (6,5)):
    if model_type=="mlp":
        # print(layer_size)
        # print(type(layer_size))
        model = MLP(in_channels = train.shape[1], hidden_channels = layer_size)
    elif model_type=="custom1":
        model = CS.Custom_Model(train.shape[1])
    elif model_type=="custom2":
        model = CS.Custom_Model2(train.shape[1])
    elif model_type=="custom3":
        model = CS.Custom_Model3(train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    return model, criterion, optimizer


# def model_obj(train, layer_size, lr):
#     NN_model = Sequential()
#     # The Input Layer :
#     NN_model.add(Dense(256, kernel_initializer='normal',input_dim = train.shape[1], activation='relu'))

#     # The Hidden Layers :
#     NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
#     NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))
#     NN_model.add(Dense(256, kernel_initializer='normal',activation='relu'))

#     # The Output Layer :
#     NN_model.add(Dense(1))

#     # Compile the network :
#     NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
#     NN_model.summary()
#     return NN_model

def run_fit(train_dataloader, model, criterion, optimizer, batch_size = 16, epoch = 200):
        for epoch in range(epoch):  # loop over the dataset multiple times
            pred = torch.empty(0)
            act = torch.empty(0)
            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                # get the inputs
                inputs, labels = data
                # logger.debug("{}: labels: = {}".format(datetime.datetime.now(), labels))
                #save the output labels in the act tensor
                act = torch.cat((act, labels))
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                output = torch.reshape(model(inputs), (1,-1))[0]
                # logger.debug("{}: outputs: = {}".format(datetime.datetime.now(), output))
                loss = criterion(output, labels)
                # print(loss)
                logger.debug("{}: loss: = {}".format(datetime.datetime.now(), loss))
                logger.debug("{}: type(loss): = {}".format(datetime.datetime.now(), type(loss)))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % batch_size == batch_size-1:    # print every 2000 mini-batches
                    logger.debug('{}: [{:d}, {:d}] loss: {:.3f}'.format(datetime.datetime.now(), epoch + 1, i + 1, running_loss / batch_size))
                    running_loss = 0.0
                #save the predicted labels in the pred tensor
                pred = torch.cat((pred, output))

            #Once this epoch is over, find the accuracy over the pred and act tensors
            metric = BinaryAccuracy()
            logger.debug("{}: Accuracy for epoch {}: = {:.3f}".format(datetime.datetime.now(), epoch+1, metric(pred, act).item()*100))
            # print("Accuracy for this epoch: %.3f"%(metric(pred, act).item()*100))



def run_evaluate(test_dataloader, model)->float:
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels = data
            logger.debug("{}: inputs: = {}".format(datetime.datetime.now(), inputs))
            logger.debug("{}: labels: = {}".format(datetime.datetime.now(), labels))
            # print("data = ", data)
            # print("inputs = ", inputs)
            # print("labels = ", labels)
            output =  torch.reshape(model(inputs), (1,-1))[0].round()
            logger.debug("{}: output: = {}".format(datetime.datetime.now(), output))
            total += labels.size(0)
            correct += (output == labels).sum().item()
    return correct/total

##################################################################################################################################################################
#utility functions
def create_test_train(dataset, test_size):
    data=pd.read_csv(dataset)
    data = data.dropna()
    dataset_name = os.path.splitext(os.path.basename(dataset))[0] #extract the name of the dataset used from the path given in dataset (no extension)
    if(dataset_name[:-1]=="dataset"):
        col_name = 'Y'
    else:
        if(dataset_name=="adult"):
            col_name = 'income'
        elif(dataset_name=="compas"):
            #categorical_encoding
            col_name = 'score'
        elif(dataset_name=="give_me_some_credit"):
            col_name = 'SeriousDlqin2yrs'
        elif(dataset_name=="heloc"):
            col_name = 'RiskPerformance'
    
    data = normalize(data, col_name) #as of now assume data is not normalized but scaled

    #save the normalized data to check
    data.to_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/normalized", dataset_name + "_model_normalized.csv"), index=False)
    
    

    if(dataset_name=="give_me_some_credit"): #Give Me Some Credit has data imbalance -> We use undersampling to deal with it
        print("Removing entries with undersampling....")
        class_count_1, class_count_0 = data[col_name].value_counts()
        # print("Initial Number of Class 0: ", class_count_0)
        # print("Initial Number of Class 1: ", class_count_1)
        # print("Length of original dataset: ", len(data.index))
        class_0 = data[data[col_name] == 0]
        # print("Length of class_0: ", len(class_0.index))
        class_1 = data[data[col_name] == 1]
        # print("Length of class_1: ", len(class_1.index))
        class_1_undersampled = class_1.sample(class_count_0)
        # print("Length of class_1_undersampled: ", len(class_1_undersampled.index))
        data = pd.concat([class_1_undersampled, class_0], axis=0)
        # print("Length of changed dataset: ", len(data.index))

    class_count_1, class_count_0 = data[col_name].value_counts()
    print("Number of Class 0 after undersampling: ", class_count_0)
    print("Number of Class 1 after undersampling: ", class_count_1)
    X=data.drop(col_name, axis=1)
    Y=data[col_name]
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=0)
    return x_train, x_test, y_train, y_test


# #min max normalization
# def normalize(dataset, col_name):
#     for i in dataset.columns:
#         if i!=col_name:
#             dataset[i] = (dataset[i]-dataset[i].min())/(dataset[i].max()-dataset[i].min())
#     return dataset

#z-score normalisation
def normalize(dataset, col_name):
    for i in dataset.columns:
        if i!=col_name:
            dataset[i] = (dataset[i]-dataset[i].mean())/(dataset[i].std())
    return dataset

# #just feature scaling
# def normalize(dataset, col_name):
#     for i in dataset.columns:
#         if i!=col_name:
#             dataset[i] = dataset[i]/1000
#     return dataset

# def encode_cat(dataset):
    

##################################################################################################################################################################
#main function
def implement_model(model_type = "mlp" , batch_size = 16, test_size = 0.3, learning_rate = 0.001, hidden_layer_size = (6,5), epoch=500):
    #paramaters

    # #generate datasets
    # while(1):
    #     try:
    #         ch = input("\nDo you want to generate the dataset?\nEnter your choice[Y\\N]: ").upper()
    #         if(ch=='Y'):
    #             dg.generateDatapoints(lower, upper)
    #         else:
    #             pass
    #         break
    #     except IOError:
    #         print("Wrong input. Please try again.")

    #choose datasets
    while(1):
        try:
            ch1 = int(input("""Choose the dataset you want:
                            \n1) dataset1.csv
                            \n2) dataset2.csv
                            \n3) dataset3.csv
                            \n4) dataset4.csv
                            \n5) dataset5.csv
                            \n6) dataset6.csv
                            \n7) dataset7.csv
                            \n8) adult.csv
                            \n9) compas.csv
                            \n10) give_me_some_credit.csv
                            \n11) heloc.csv
                            \nEnter your choice: """))
            if(ch1<1 or ch1>11):
                print("Incorrect Input.")
            else:
                
                if(ch1==1):
                    dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "dataset1.csv")
                elif(ch1==2):
                    dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "dataset2.csv")
                elif(ch1==3):
                    dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "dataset3.csv")
                elif(ch1==4):
                    dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "dataset4.csv")
                elif(ch1==5):
                    dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "dataset5.csv")
                elif(ch1==6):
                    dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "dataset6.csv")
                elif(ch1==7):
                    dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "dataset7.csv")
                elif(ch1==8):
                    dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "adult.csv")
                elif(ch1==9):
                    dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "compas.csv")
                elif(ch1==10):
                    dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "give_me_some_credit.csv")
                elif(ch1==11):
                    dataset = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "heloc.csv")
            break
        except IOError:
            print("Wrong input. Please try again.")

    #training split using sklearn
    x_train, x_test, y_train, y_test = create_test_train(dataset, test_size) #returns dataframes
    #convert to tensors
    x_train_tensor = torch.from_numpy(x_train.values).to(torch.float32) #dataframe.values gives a numpy array
    x_test_tensor = torch.from_numpy(x_test.values).to(torch.float32)
    y_train_tensor = torch.from_numpy(y_train.values).to(torch.float32)
    y_test_tensor = torch.from_numpy(y_test.values).to(torch.float32) #labels must be short
    #create tensoir datasets
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    #load dataloaders
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size)

    model, criterion, optimizer = model_maker(model_type, x_train, learning_rate, hidden_layer_size)
    run_fit(train_dataloader, model, criterion, optimizer, batch_size, epoch)

    # Calcuate test accuracy
    print("Checking for test data... ")
    score = run_evaluate(test_dataloader, model)
    print("Test Accuracy: %.3f"%(score*100))
    logger.debug("{}: Test Accuracy: {:.3f}".format(datetime.datetime.now(), score*100))
    
    #choose to save model
    while(1):
        try:
            ch2 = input("Do you want to save the model?\nEnter your choice[Y\\N]: ").upper()
            if(ch2=='Y'):
                if(ch1==1):
                    if(model_type == "mlp"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "mlp_classifier_model_D1.pt")
                    elif (model_type == "custom1"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential1_model_D1.pt")
                    elif (model_type == "custom2"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential2_model_D1.pt")
                    elif (model_type == "custom3"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential3_model_D1.pt")
                elif(ch1==2):
                    if(model_type == "mlp"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "mlp_classifier_model_D2.pt")
                    elif (model_type == "custom1"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential1_model_D2.pt")
                    elif (model_type == "custom2"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential2_model_D2.pt")
                    elif (model_type == "custom3"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential3_model_D2.pt")
                elif(ch1==3):
                    if(model_type == "mlp"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "mlp_classifier_model_D3.pt")
                    elif (model_type == "custom1"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential1_model_D3.pt")
                    elif (model_type == "custom2"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential2_model_D3.pt")
                    elif (model_type == "custom3"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential3_model_D3.pt")
                elif(ch1==4):
                    if(model_type == "mlp"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "mlp_classifier_model_D4.pt")
                    elif (model_type == "custom1"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential1_model_D4.pt")
                    elif (model_type == "custom2"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential2_model_D4.pt")
                    elif (model_type == "custom3"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential3_model_D4.pt")
                elif(ch1==5):
                    if(model_type == "mlp"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "mlp_classifier_model_D5.pt")
                    elif (model_type == "custom1"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential1_model_D5.pt")
                    elif (model_type == "custom2"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential2_model_D5.pt")
                    elif (model_type == "custom3"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential3_model_D5.pt")
                elif(ch1==6):
                    if(model_type == "mlp"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "mlp_classifier_model_D6.pt")
                    elif (model_type == "custom1"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential1_model_D6.pt")
                    elif (model_type == "custom2"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential2_model_D6.pt")
                    elif (model_type == "custom3"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential3_model_D6.pt")
                elif(ch1==7):
                    if(model_type == "mlp"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "mlp_classifier_model_D7.pt")
                    elif (model_type == "custom1"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential1_model_D7.pt")
                    elif (model_type == "custom2"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential2_model_D7.pt")
                    elif (model_type == "custom3"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential3_model_D7.pt")
                elif(ch1==8):
                    if(model_type == "mlp"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "mlp_classifier_model_D8.pt")
                    elif (model_type == "custom1"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential1_model_D8.pt")
                    elif (model_type == "custom2"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential2_model_D8.pt")
                    elif (model_type == "custom3"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential3_model_D8.pt")
                elif(ch1==9):
                    if(model_type == "mlp"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "mlp_classifier_model_D9.pt")
                    elif (model_type == "custom1"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential1_model_D9.pt")
                    elif (model_type == "custom2"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential2_model_D9.pt")
                    elif (model_type == "custom3"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential3_model_D9.pt")
                elif(ch1==10):
                    if(model_type == "mlp"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "mlp_classifier_model_D10.pt")
                    elif (model_type == "custom1"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential1_model_D10.pt")
                    elif (model_type == "custom2"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential2_model_D10.pt")
                    elif (model_type == "custom3"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential3_model_D10.pt")
                elif(ch1==11):
                    if(model_type == "mlp"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "mlp_classifier_model_D11.pt")
                    elif (model_type == "custom1"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential1_model_D11.pt")
                    elif (model_type == "custom2"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential2_model_D11.pt")
                    elif (model_type == "custom3"):
                        model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../trained_models", "custom_sequential3_model_D11.pt")
                model_scripted = torch.jit.script(model)# Export to TorchScript
                model_scripted.save(model_save_path) # Save
            else:
                pass
            break
        except IOError:
            print("Wrong input. Please try again.")
    

batch_size = 16
test_size = 0.3
learning_rate = 0.01
hidden_layer_size = (6,5)
epochs=1000
inp = input("""Enter the model type you want:
            \n1.Custom Sequential 1
            \n2.Custom Sequential 2
            \n3.Custom Sequential 3
            \n4.MLP
            \nEnter your choice: """)
model_type = ["custom1", "custom2", "custom3", "mlp"]
implement_model(model_type[int(inp)-1], batch_size, test_size, learning_rate, hidden_layer_size, epochs)