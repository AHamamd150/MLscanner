# Exploration of Parameter Spaces Assisted by Machine Learning

 &emsp; Scanner package assisted by  Machine learning models for faster convergence to the target area in high diemnsions. The packge based on [arXiv:2207.09959 [hep-ph]](https://arxiv.org/abs/2207.09959). 
## Introduction
&emsp;  In this package we have implemented two broad classes of ML based efficient sampling methods of parameter spaces, using regression and classification. we employ an iterative process where the ML model is trained on a set of parameter values and their results from a costly calculation, with the same ML model later used to predict the location of additional relevant parameter values. The ML model is refined in every step of the process, therefore, increasing the accuracy of the regions it suggests. We attempt to develop a generic tool that can take advantage of the improvements brought by this iterative process. we set the goal in filling the regions of interest such that in the end we provide a sample of parameter values that densely spans the region as requested by the user. With enough points sampled, it should not be difficult to proceed with more detailed studies on the
implications of the calculated observables and the validity of the model under question. We pay special attention to give control to the user over the many hyperparameters involved in the process, such as the number of nodes, learning rate, training epochs, among many others, while also suggesting defaults that work in many cases. The user has the freedom to decide whether to use regression or classification to sample the parameter space, depending mostly on the complexity of the problem. For example, with complicated fast changing likelihoods it may be easier for a ML classifier to converge and suggest points that are likely inside the region of interest. However, with observables that are relatively easy to adjust, a ML regressor may provide information useful
to locate points of interest such as best fit points, or to estimate the distribution of the parameters. After several steps in the iterative process, it is expected to have a ML model that can accurately predict the relevance of a parameter point much faster than passing through the complicated time
consuming calculation that was used during training. As a caveat, considering that this process requires iteratively training a ML model during several epochs, which also requires time by itself, for cases where the calculations can be optimized to run rather fast, other methods may actually
provide good results in less time.

## Requirements
&emsp; To run the package you need python3 with the following modules:
* Numpy
* TensorFlow
* sklearn
* imblearn 
* multiprocessing (for the intial training over multi-cores)
* tqdm (for the illustration of the fancy progress bar)

These packages can be easily installed by `pip3 install module`

## Package structure
&emsp; The package consists of the following:
* `run.sh` shall script that used to excute the package python files 
* `scan_input.py` input file that the user has to fill it. The user can control the run via the switches in this file
* `ML_regressor_genericFunctions.ipynb` google colab notebook that inclide the scan over the generic fucntions. The user can use it to scan over defined function. The class `scan()` include the following ML models:
  * DNNR: MLP regressor with 4 hidden layers, 100 nueron each and MSE loss function.
  * GBR : GradientBoostingRegressor
  * RFR : RandomForestRegressor
  * SVMRBF: Supported vector regressor wiht RBF kernel
  * SVMPOLY: Supported vector regressor wiht polynomial kernel
* `docs/` directory include the following:
  * `Install` documentary on how to install the package
  * `Run the package` documentay on how to run the package
  * `how to adjust the input file` documentray on how the user adjust the input file
* `source/` directory inculde the following source files:
  * `auxiliary.py` file include the auxiliary functions to link spheno with HB/HS and functions for parallel run
  * `MLs_HEP.py`   main file with the scanner loop. The class `scan()` is used to access the type of the needed ML 

## Get started
&emsp; To run the package:
* Download and extract the packge in your local PC
* Spheno, HiggsBounds and HiggsSignals must be installed individually
* `chmod 777 run.sh`
* `scan_input.py` must be adjust by the user
* `./run.sh ML` with ML is the name of one of the implemented ML models, e.g. DNNR for MLP regressor or DNNC for MLP classifier, etc
* After the scan finished an output directory called `result` will be created in the same package directory contains the following 
  * File conatins the accumlated points file 
  * File contains the corresponding chi squared values from HiggsSignals
  * ML model saved weights to be used for the future without further taining
## $$\textcolor{red}{\text{Animation to demonstrate how the ML can suggest points in the target region.}}$$ 
The 2d and 3d functions are defined as: 

&emsp;&emsp;&emsp; $F_{2d} = [2+\cos\frac{x_1}{5}\cos\frac{x_2}{7}]^5$ &emsp; & &emsp; $F_{3d} = [2+\cos\frac{x_1}{7}\cos\frac{x_2}{7}\cos\frac{x_3}{7}]^5$
 
The animation shows how the RandomForest regressor  is used to speed up the scan convergence to $F_{2d/3d}= 100$ with standard diviation of 20. The MAE metric is used to determine the convergence after each iteration.

https://user-images.githubusercontent.com/68282212/175757994-dda3b29f-61fb-45f4-b8ed-c055f9430723.MOV

https://user-images.githubusercontent.com/68282212/175757998-81b09c8f-3b2d-4869-8dec-e42a4b6c1599.MOV



