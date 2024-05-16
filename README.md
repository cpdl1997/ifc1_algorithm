# IFC1: A novel algorithm to learn user preferences and generate counterfactuals based on it

This is a repository for **IFC1**, a novel algorithm to learn user preferences that is based on the duelling bandit IF1 by Ue et al. [^1].

## Setup details:

The code has been written in python 3.11.7 and pytorch 2.2.0+cu121. The remaining requirements are all provided in the requirements.txt which can be used to install via the following commands:
* *Windows*: `pip install -r requirements.txt`
* *Linux*: `pip install -r requirements.txt`

To create the environment 'alg4' under the folder 'virtual_environment', we use (make sure that python 3.11 installed in system from before):
* *Windows*: `py -3.11 -m venv .\virtual_environment\alg4`
* *Linux*: `python3.11 -m venv .\virtual_environment\alg4`

While using conda, you dont even need to have python 3.11 installed from before. We can use:
* *Windows and Linux*: `conda create -n .\virtual_environment\alg4 python=3.11`
When creating with conda, feel free to use pip therafter to install other packages - this is necessary as causal-learn and dowhy are not available on conda.

---
> [!NOTE]

<blockquote>

It is recommended that you comment out the following packages first in your requirements.txt:
1. tensorflow:
   - tensorboard==2.15.1
   - tensorboard-data-server==0.7.2
   - tensorflow==2.15.0
   - tensorflow-estimator==2.15.0
   - tensorflow-intel==2.15.0
   - tensorflow-io-gcs-filesystem==0.31.0
2. pytorch:
   - torch==2.2.0+cu121
   - torchaudio==2.2.0+cu121
   - torchmetrics==1.3.0.post0
   - torchvision==0.17.0+cu121

Thereafter install tensorflow simply by using: `pip install tensorflow`

Install pytorch using the official site for [current version](https://pytorch.org/get-started/locally/) or [older versions](https://pytorch.org/get-started/previous-versions/).
</blockquote>
---

## How to use:

To run the baselines for the synthetic dataset, run the following command:
* *Windows*: `python .\src\baselines\compare2.py`

To run the baselines for the dataset 'Give Me Some Credit', run the following command:
* *Windows*: `python .\src\baselines\alg10_multiple_run.py`

Make sure that folder directory names mentioned within the file are correct and valid for you. The logs will be available in the 'logs' folder under the name 'Log_comparison2.log' and 'Log_give_me_credit_alg10.log' respectively. The graphs will be available in the 'results\graphs' folder, with each user getting a different folder for compare2.py, and the whole run getting two folders for alg10_multiple_run.py.

If we want to run the individual files for a single user, try running the following command:
* *Windows*: `python .\src\counterfactual_gen\Algorithm10.py`


## About the code:

### `counterfactual_gen` module
**Algorithm10.py** contains the actual IFC1 algorithm which has been used for the final results. The other files contain previous versions of the algorithm and are variations of it. The other important baselines that have been used are:
1. ***manan_algorithmic_recourse2.py*** -> Manan's base paper (Version 2)
2. ***manan_with_causality_algorithmic_recourse2.py*** -> Manan's base paper with causality involved (Version 2)
3. ***my_method_algorithmic_recourse.py*** -> Custom 1
4. ***my_method_algorithmic_recourse3.py*** -> Custom 2

Some of the other files have been shifted to a separate folder and are described as below:
* ***manan_algorithmic_recourse.py*** -> Original implementation of Manan's base paper which was later modified in version 2 (see above) to get better results and more optimized outcomes.
* ***manan_with_causality_algorithmic_recourse.py*** -> Original implementation of Manan's base paper but with added causality changes, which was later modified in version 2 (see above) to get better results and more optimized outcomes.
* ***my_method_algorithmic_recourse2.py*** -> Some older implementation of Custom 2
* ***my_method_algorithmic_recourse4.py*** -> First attempt at involving causality in Custom 2 (which is basically our actual method)
* ***Algorithm5.py***-***Algorithm9.py*** -> Older implementations of our full workflow, with some not involving the full causality module with causal-learn and dowhy possibly. As we approach Algorithm9 we get closer to the final full version Algorithm10.py. Algorithm10.py has the final optimized version of Algorithm9 (along with sone metric fixes).

### `algorithms` module
This folder contains the algorithms to generate counterfactuals. **TenthMethod.py** contains the final method to generate counterfactuals. 
The other important baselines that have been used are:
1. ***WachterCF.py*** -> Manan's base paper uses this - this is the original Wachter implementation of the code along with the optimized version
2. ***MananWithCausalityCF.py*** -> Manan's base paper with causality uses this along with the optimized version
3. ***baseCF.py*** -> Base class which is the basis for all the methods

Some of the other files have been shifted to a separate folder and are described as below:
* ***FirstMethod.py***-***NinthMethod.py*** -> Older implementations of 'TenthMethod.py'. Some older ones dont have the proper causality model, some in between are unoptimized for causality (not involving the use of the dictionary) and some near the end are just similar files but used by the corresponding files Algorithm5-Algorithm9.

### `dataset_gen` module
* ***dataset_generator.py*** -> Used to create the synthetic datasets and save as csv files
* ***dataset_pickler.py*** -> Used to pickle the datasets - it may or may not be working as it was tested a long time ago

### `model` module
* ***create_model.py*** -> Used to create the models we use to run our method using custom_sequential.py
* ***custom_sequential.py*** -> Model definitions are available here
* ***mlp_classifier.py*** -> Model definitions of a simple MLP model - it may not be working as of now

### `scm` module
* ***Node.py*** and ***SCM.py***-> Used to create the custom SCM models we use with MananWithCausalityCF.py
This folder also contains causallearn, dowhy and graphviz packages in its entirety - this was done because there were some issues with respect to imports.

### `util` module
This folder contains two different data structures with specific functionalities involved in Node.py and SCM.py


## Changes for further versions:

There is a need to further optimize the code, introduce GPU functionalities and multithreading, and cleaning up some of the wasted files.


[^1]: Yue, Yisong, et al. "The k-armed dueling bandits problem." Journal of Computer and System Sciences 78.5 (2012): 1538-1556.

