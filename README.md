# IFC1: A novel algorithm to learn user preferences and generate counterfactuals based on it

This is a repository for **IFC1**, a novel algorithm to learn user preferences that is based on the duelling bandit IF1 by Ue et al. [^1].

## Setup details:

The code has been written in python 3.11.7 and pytorch 2.2.0+cu121. The remaining requirements are all provided in the requirements.txt which can be used to install via the following commands:
    *Windows: `pip install -r requirements.txt`
    *Linux: `pip install -r requirements.txt`

To create the environment 'alg4' under the folder 'virtual_environment', we use (make sure that python 3.11 installed in system from before):
    *Windows: `py -3.11 -m venv .\virtual_environment\alg4`
    *Linux: `python3.11 -m venv .\virtual_environment\alg4`
While using conda, you dont even need to have python 3.11 installed from before. We can use:
    *Windows and Linux: `conda create -n .\virtual_environment\alg4 python=3.11`
When creating with conda, feel free to use pip therafter to install other packages - this is necessary as causal-learn and dowhy are not available on conda.

## How to use:

To run the baselines for the synthetic dataset, run the following command:
    *Windows: `python .\src\baselines\compare2.py`
To run the baselines for the dataset 'Give Me Some Credit', run the following command:
    *Windows: `python .\src\baselines\alg10_multiple_run.py`
Make sure that folder directory names mentioned within the file are correct and valid for you. The logs will be available in the 'logs' folder under the name 'Log_comparison2.log' and 'Log_give_me_credit_alg10.log' respectively. The graphs will be available in the 'results\graphs' folder, with each user getting a different folder for compare2, and the whole run getting two folders for alg10_multiple_run.

If we want to run the individual files for a single user, try running the following command:
    *Windows: `python .\src\counterfactual_gen\Algorithm10.py`









[^1]: Yue, Yisong, et al. "The k-armed dueling bandits problem." Journal of Computer and System Sciences 78.5 (2012): 1538-1556.

