import os
import random
import pandas as pd

def pickleDataset():
    ch = input("Enter which dataset you would like to pickle:\n1) Dataset 1\n2) Dataset 2\n3) Dataset 3\nEnter your choice: ")
    if(ch!='1' and ch!='2' and ch!='3'):
        print("Incorrect Input. Restarting...")
        pickleDataset()
    else:
        ch=int(ch)
        if(ch==1):
            dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "dataset1.csv")
        elif(ch==2):
            dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "dataset2.csv")
        else:
            dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "dataset3.csv")

        dataset = pd.read_csv(dataset_path)

        if(ch==1):
            pickle_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/pickle", "dataset1.pkl")
        elif(ch==2):
            pickle_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/pickle", "dataset2.pkl")
        else:
            pickle_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset/pickle", "dataset3.pkl")

        dataset.to_pickle(pickle_path)
        print('Dataset Pickled Successfully.')
    return
