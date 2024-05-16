import pandas as pd
import numpy as np
import os
import sys

user_choice_subset = [4]
b_hat=[3]
b=[4]
score1 = 0
score2 = 0
for i in user_choice_subset: #if any feature in user's preferred subset lies in any of the options (b_hat and b), then their score increases
    if i in b_hat:
        score1 = score1 + 1
    if i in b:
        score2 = score2 + 1
if score1>score2: #Choose the one which matches user's preferred subset most
    ch=1
elif score1<score2:
    ch=2
else: #in case of match, choose the shorter option
    if len(b_hat)>len(b):
        ch = 2
    else:
        ch = 1
print("Logically chosen choice is Option {}.".format(ch))