# IFC1: A novel algorithm to learn user preferences and generate counterfactuals based on it

This is a repository for **IFC1**, a novel algorithm to learn user preferences that is based on the duelling bandit IF1 by Ue et al. [^1]. The code has been written in python 3.11.7 and pytorch 2.2.0+cu121. The remaining requirements are all provided in the requirements.txt which can be used to install via the following commands:
    * Windows: `pip install -r requirements.txt`
    * Linux: `pip install -r requirements.txt`

To create the environment 'alg4' under the folder 'virtual_environment', we use (make sure that python 3.11 installed in system from before):
    * Windows: `py -3.11 -m venv .\virtual_environment\alg4`
    * Linux: `python3.11 -m venv .\virtual_environment\alg4`
While using conda, you dont even need to have python 3.11 installed from before. We can use:
    * Windows and Linux: `conda create -n .\virtual_environment\alg4 python=3.11`
When creating with conda, feel free to use pip therafter to install other packages - this is necessary as causal-learn and dowhy are not available on conda.

[^1]: Yue, Yisong, et al. "The k-armed dueling bandits problem." Journal of Computer and System Sciences 78.5 (2012): 1538-1556.

