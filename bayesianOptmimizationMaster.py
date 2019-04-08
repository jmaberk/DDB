'''
Name: bayesianOptimizationMaster.py
Authors: Julian Berk and Vu Nguyen
Publication date:16/04/2018
Inputs:None
Outputs: Pickle files and plots containing the results from experiments run
Description: The master file for code used to generate the results for the
paper Exploration Enhanced Expected Improvement for Bayesian Optimization.
Most aspects of the algorithm can be altered from this file. See comments for
more details
'''
###############################################################################
import sys
sys.path.insert(0,'../../')
from prada_bayes_opt import PradaBayOptFn
import numpy as np
from prada_bayes_opt import auxiliary_functions
from prada_bayes_opt import functions
from prada_bayes_opt import real_experiment_function
from prada_bayes_opt.utility import export_results
import plot_results
import pickle
import random
import time
import matplotlib.pyplot as plt
#import pickle
import warnings
import itertools
warnings.filterwarnings("ignore")

'''
***********************************IMPORTANT***********************************
The pickle_location variable below must be changed to the appropriate directory
in your system for the code to work.
'''
pickle_location="D:\OneDrive\Documents\PhD\Code\Bayesian\DDB\pickleStorage"
###############################################################################
'''
Here the user can choose which functions to optimize. Just un-comment the 
desired functions and set the desired dimensions with the dim parameter
in supported functions
'''
###############################################################################
myfunction_list=[]
#myfunction_list.append(functions.schwefel(dim=1))
#myfunction_list.append(functions.doubleGaussian(dim=1))
#myfunction_list.append(functions.gaussianMixtureExt(dim=1))
#myfunction_list.append(functions.gaussian(dim=8))
#myfunction_list.append(functions.mixture(peaks=3))
#myfunction_list.append(functions.beale())
#myfunction_list.append(functions.forrester())
#myfunction_list.append(functions.franke())
#myfunction_list.append(functions.rosenbrock())
#myfunction_list.append(functions.eggholder())
#myfunction_list.append(functions.shubert())
#myfunction_list.append(functions.schwefel(dim=4))
#myfunction_list.append(functions.griewank(dim=3))
myfunction_list.append(functions.branin())
#myfunction_list.append(functions.dropwave())
#myfunction_list.append(functions.sixhumpcamel())
#myfunction_list.append(functions.hartmann_3d())
#myfunction_list.append(functions.levy(dim=5))
#myfunction_list.append(functions.ackley(input_dim=5))
#myfunction_list.append(functions.alpine1(input_dim=5))
#myfunction_list.append(functions.alpine2(input_dim=5))
#myfunction_list.append(functions.hartmann_6d())
#myfunction_list.append(functions.alpine2(input_dim=10))
#myfunction_list.append(functions.gSobol(a=np.array([1,1,1,1,1,1,1,1,1,1])))
#myfunction_list.append(functions.gSobol(a=np.array([1,1,1,1,1,1,1,1,1,1,1,1])))
#myfunction_list.append(functions.gSobol(a=np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])))
#myfunction_list.append(real_experiment_function.SVR_function())
#myfunction_list.append(real_experiment_function.AlloyCooking_Profiling())
#myfunction_list.append(real_experiment_function.AlloyCooking_Profiling_3Steps())
#myfunction_list.append(real_experiment_function.Robot_BipedWalker())
#myfunction_list.append(real_experiment_function.DeepLearning_MLP_MNIST())
#myfunction_list.append(real_experiment_function.DeepLearning_MLP_MNIST_5D())
#myfunction_list.append(real_experiment_function.BayesNonMultilabelClassification())
#myfunction_list.append(real_experiment_function.BayesNonMultilabelClassificationEnron())

###############################################################################
'''
Here the user can choose which acquisition functions will be used. To select
an acquisition function, un-comment the "acq_type_list.append(temp)" after its
name. If you do not have any pickle files for the method and function, you will
also need to comment out the relevent section in plot_results.py.
'''
###############################################################################
acq_type_list=[]

temp={}
temp['name']='ei'
acq_type_list.append(temp)

temp={}
temp['name']='ei_regularizerH'
#acq_type_list.append(temp)

mybatch_type_list={'Single'}
###############################################################################
'''
Here the user can choose the type of search bound. To select
a boundary method, un-comment the "bound_type_list.append(temp)" after its
name. If you do not have any pickle files for the method and function, you will
also need to comment out the relevent section in plot_results.py.
'''
###############################################################################
bound_type_list=[]                                                       

temp={}
temp='fixed'
#bound_type_list.append(temp)

temp={}
temp='expandBoundsDDB_MAP'
bound_type_list.append(temp)

temp={}
temp='expandBoundsDDB_FB'
bound_type_list.append(temp)

temp={}
temp='volumeDoubling'
#bound_type_list.append(temp)

temp={}
temp='expandBoundsFiltering'
#bound_type_list.append(temp)

###############################################################################
'''
#1 seed is used along with the experiment number as a seed to randomly generate
the initial points. Setting this as a constant will allow results to be
reproduced while making it random will let each set of runs use a different
set of initial points.
#2 num_initial_points controls the number of random sampled points each 
experiment will start with.
#3 max_iterations controls the number of iterations of Bayesian optimization
that will run on the function. This must be controlled with iteration_factor
for compatability with the print function.
#4 num_repeats controls the number of repeat experiments.
5# acq_params['optimize_gp'] If this is 1, then the lengthscale will be
determined by maximum likelihood every 15 samples. If any other value, no
lengthscale adjustement will be made
6# random_initial_bound controls the inituial bound. This is set a a box of
length 1 in each direction. If random_initial_bound=1 then it is placed 
randomly in the maximum search space.
'''
###############################################################################
seed=1
#seed=np.random.randint(1,100) #1
print("Seed of {} used".format(seed))

for idx, (myfunction,acq_type,mybatch_type,) in enumerate(itertools.product(myfunction_list,acq_type_list,mybatch_type_list)):
    func=myfunction.func
    mybound=myfunction.bounds
    yoptimal=myfunction.fmin*myfunction.ismax
    
    acq_type['dim']=myfunction.input_dim 
    
    num_initial_points=3*myfunction.input_dim+1 #2
    
    iteration_factor=20 #3
    max_iterations=iteration_factor*myfunction.input_dim 
    
    num_repeats=20 #4
    
    GAP=[0]*num_repeats
    ybest=[0]*num_repeats
    Regret=[0]*num_repeats
    MyTime=[0]*num_repeats
    MyOptTime=[0]*num_repeats
    ystars=[0]*num_repeats

    func_params={}
    func_params['bounds']=myfunction.bounds
    func_params['f']=func

    acq_params={}
    acq_params["bb_function"]=myfunction
    acq_params['iteration_factor']=iteration_factor
    acq_params['acq_func']=acq_type
    acq_params['optimize_gp']=1 #5if 1 then maximum likelihood lenghscale selection will be used
    acq_params['random_initial_bound']=1 #6 if 1 then the initial bound will be chosen at random
    for bound in bound_type_list:
        acq_params['expandSS']=bound
    
        for ii in range(num_repeats):
            
            gp_params = {'theta':0.1,'noise_delta':0.1} # Kernel parameters for the square exponential kernel
            baysOpt=PradaBayOptFn(gp_params,func_params,acq_params,experiment_num=ii,seed=seed)
    
            ybest[ii],MyTime[ii]=auxiliary_functions.run_experiment(baysOpt,gp_params,
                                                    yoptimal,n_init=num_initial_points,NN=max_iterations)
                                                          
            MyOptTime[ii]=baysOpt.time_opt
            ystars[ii]=baysOpt.ystars
        Score={}
        Score["GAP"]=GAP
        Score["ybest"]=ybest
        Score["ystars"]=ystars
        Score["Regret"]=Regret
        Score["MyTime"]=MyTime
        Score["MyOptTime"]=MyOptTime
        Score['expandSS']=acq_params['expandSS']
        export_results.print_result_ystars(baysOpt,myfunction,Score,mybatch_type,acq_type,acq_params,toolbox='PradaBO')

#Plots the results. Comment out to supress plots.
for idx, (myfunction) in enumerate(itertools.product(myfunction_list)):
    plot_results.plot(myfunction[0].name,myfunction[0].input_dim,iteration_factor,pickle_location)    