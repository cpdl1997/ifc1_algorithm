import math
import numpy as np
import os
import random
from scipy.special import expit

##################################################################################################################################################################
#actual functions to be found with no causality
def actual_function1(inp1, inp2):
    val = 2*(inp1**2)+inp2*5 + 4
    if val>350:
        return 1
    return 0



def actual_function2(inp1, inp2):
    val = 1/(inp1**2+math.sqrt(abs(inp2))*3+1) + math.sin(3*inp1*inp2+5)
    if val >= 0.5:
        return 1
    return 0



def actual_function3(inp1, inp2, inp3, inp4, inp5):
    # val = expit((5/(abs(inp1**abs((inp2)))+1) + np.random.normal(inp3, abs(inp4)))*inp5)
    val = np.exp(-np.abs(inp1+inp2+inp3+inp4+inp5))+ np.random.normal(0, 1)
    if val>=0.5:
        return 1
    return 0


def actual_function7(inp1, inp2, inp3, inp4, inp5):
    val = expit(inp1+inp2+inp3+inp4+inp5)
    if val>=0.5:
        return 1
    return 0

##################################################################################################################################################################
#actual functions to be found with causality involved
def actual_function4(inp1, inp2, inp3, inp4, inp5):
    val = expit(inp1+inp2+inp3+inp4+inp5)
    if val>=0.5:
        return 1
    return 0



def actual_function5(inp1, inp2):
    val = 1/(inp1**2+math.sqrt(abs(inp2))*3+1) + math.sin(3*inp1*inp2+5)
    if val >= 0.5:
        return 1
    return 0



def actual_function6(inp1, inp2, inp3, inp4, inp5):
    val = expit((5/(abs(inp1**abs((inp2)))+1) + np.random.normal(inp3, abs(inp4)))*inp5)
    if val>=0.5:
        return 1
    return 0
##################################################################################################################################################################
#generate datapoints for testing counterfactuals
def generateDatapoints(lower=-100, upper=100):
    ch = int(input("""Enter which function you would like to generate a dataset for:
               \n1) Simple Polynomial Function in two variable
               \n2) Complex Function in two variable
               \n3) Complex Function in 5 variables with Gaussian Noise
               \n4) Simple Function in 5 variables with causality
               \n5) Complex Function in two variable with causality
               \n6) Complex Function in 5 variables with Gaussian Noise with causality
               \n7) Simple Function in 5 variables
               \nEnter your choice: """))
    
    if(ch<1 or ch>7):
        print("Incorrect Input. Restarting...")
        generateDatapoints(lower, upper)
    else:
        n = int(input("Enter the number of datapoints: "))
        
        if(ch==1):
            #remove file if it exists
            try:
                os.remove(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "dataset1.csv"))
            except OSError:
                pass
            #create file
            f = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "dataset1.csv"), 'w')
            f.write('X1,X2,Y\n')
            for _ in range(0,n):
                inp1 = random.randint(lower, upper)
                inp2 = random.randint(lower, upper)
                f.write(str(inp1)+','+str(inp2)+','+str(actual_function1(inp1, inp2))+'\n')
            f.close()
        elif(ch==2):
            #remove file if it exists
            try:
                os.remove(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "dataset2.csv"))
            except OSError:
                pass
            #create file
            f = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "dataset2.csv"), 'w')
            f.write('X1,X2,Y\n')
            for _ in range(0,n):
                inp1 = random.randint(lower, upper)
                inp2 = random.randint(lower, upper)
                f.write(str(inp1)+','+str(inp2)+','+str(actual_function2(inp1, inp2))+'\n')
            f.close()
        elif(ch==3):
            #remove file if it exists
            try:
                os.remove(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "dataset3.csv"))
            except OSError:
                pass
            #create file
            f = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "dataset3.csv"), 'w')
            f.write('X1,X2,X3,X4,X5,Y\n')
            for _ in range(0,n):
                inp1 = random.randint(lower, upper)
                inp2 = random.randint(lower, upper)
                inp3 = random.randint(lower, upper)
                inp4 = random.randint(lower, upper)
                inp5 = random.randint(lower, upper)
                f.write(str(inp1)+','+str(inp2)+','+str(inp3)+','+str(inp4)+','+str(inp5)+','+str(actual_function3(inp1, inp2, inp3, inp4, inp5))+'\n')
            f.close()
        elif(ch==4):
            #remove file if it exists
            try:
                os.remove(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "dataset4.csv"))
            except OSError:
                pass
            #create file
            f = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "dataset4.csv"), 'w')
            f.write('X1,X2,X3,X4,X5,Y\n')
            for _ in range(0,n):
                u1, u2, u3, u4, u5 = np.random.randint(int(lower*0.7),int(upper*0.7), size=5)
                # print("{}, {}, {}, {}, {}".format(u1, u2, u3, u4, u5))
                inp1 = u1
                inp2 = 2*inp1  + u2
                inp3 = 3*inp1  + u3
                inp4 = inp2 - 2*inp3  + u4
                inp5 = 2*inp3 - inp1  + u5
                f.write(str(inp1)+','+str(inp2)+','+str(inp3)+','+str(inp4)+','+str(inp5)+','+str(actual_function7(inp1, inp2, inp3, inp4, inp5))+'\n')
            f.close()
        elif(ch==5):
            #remove file if it exists
            try:
                os.remove(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "dataset5.csv"))
            except OSError:
                pass
            #create file
            f = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "dataset5.csv"), 'w')
            f.write('X1,X2,Y\n')
            for _ in range(0,n):
                inp1 = random.randint(lower, upper)
                inp2 = random.randint(lower, upper)
                f.write(str(inp1)+','+str(inp2)+','+str(actual_function5(inp1, inp2))+'\n')
            f.close()
        
        elif(ch==6):
            #remove file if it exists
            try:
                os.remove(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "dataset6.csv"))
            except OSError:
                pass
            #create file
            f = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "dataset6.csv"), 'w')
            f.write('X1,X2,X3,X4,X5,Y\n')
            for _ in range(0,n):
                inp1 = random.randint(lower, upper)
                inp2 = random.randint(lower, upper)
                inp3 = random.randint(lower, upper)
                inp4 = random.randint(lower, upper)
                inp5 = random.randint(lower, upper)
                f.write(str(inp1)+','+str(inp2)+','+str(inp3)+','+str(inp4)+','+str(inp5)+','+str(actual_function6(inp1, inp2, inp3, inp4, inp5))+'\n')
            f.close()

        elif(ch==7):
            #remove file if it exists
            try:
                os.remove(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "dataset7.csv"))
            except OSError:
                pass
            #create file
            f = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../dataset", "dataset7.csv"), 'w')
            f.write('X1,X2,X3,X4,X5,Y\n')
            for _ in range(0,n):
                inp1 = random.randint(lower, upper)
                inp2 = random.randint(lower, upper)
                inp3 = random.randint(lower, upper)
                inp4 = random.randint(lower, upper)
                inp5 = random.randint(lower, upper)
                f.write(str(inp1)+','+str(inp2)+','+str(inp3)+','+str(inp4)+','+str(inp5)+','+str(actual_function7(inp1, inp2, inp3, inp4, inp5))+'\n')
            f.close()
        
        print('Dataset Generated Successfully.')
    return str(ch)

if __name__=="__main__":
    generateDatapoints()

