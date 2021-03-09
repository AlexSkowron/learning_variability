from smc_object import smc_object
import numpy
import os
import sys

sub=sys.argv[1]
par=sys.argv[2]
par_set=sys.argv[3]
 
ID = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30']

if par == 1: # alpha chosen
    par_simul=numpy.arange(start=0, stop=1, step=0.1)
    param_sim=numpy.array(par_simul[par_set], 0.3, 20)
elif par == 2:
    par_simul=numpy.arange(start=0, stop=1, step=0.1)
    param_sim=numpy.array(0.8, par_simul[par_set], 20)
elif par == 3: # beta
    par_simul=numpy.arange(start=0.1, stop=100.1, step=10)
    param_sim=numpy.array(0.8, 0.3, par_simul[par_set])
else
    raise ValueError('No simulation was done')
    
print('starting sim_sub-' + sub + '_par' + par + '_ite' + par_set)

RootDir = '/home/mpib/skowron/aging_learning_var/learning_variability/'
PayOffDir = '/Users/skowron/Volumes/tardis/skowron/deBoer_RL/'

show_prg=False # show progress

reward_probs=numpy.genfromtxt(PayOffDir + 'PayOff.csv', delimiter=',') #experimental payoff scheme    
reward_probs = reward_probs.astype('float')

if os.path.isdir(RootDir + 'model_fit/results/noiselessRL/') == 0:
    os.makedirs(RootDir + 'model_fit/results/noiselessRL/')

'''
load data
'''

sub_ind = sub + '_' + par + '_' + par_set

# simulate rewards for recovery
rewards_sim=numpy.empty((2,reward_probs.shape[1]))
rewards_sim[:]=numpy.nan

for r in range(reward_probs.shape[1]):
    
    if numpy.random.rand() < reward_probs[0,r]:
        rewards_sim[0,r]=1
    else: 
        rewards_sim[0,r]=0
    
    if numpy.random.rand() < reward_probs[1,r]:
        rewards_sim[1,r]=1
    else: 
        rewards_sim[1,r]=0

rewards_sim = rewards_sim.astype('float')

'''
---
1. A dictionary with keys
 	1. *actions*, of shape (T) speciying the actions of the subject - (0 or 1). T is the total number of trials
 	1. *rewards*, of shape (2, T) with T the length of the experiment. These should be normalised between 0 and 1. In the partial case, the learning rule of the unchosen option will override the unchosen reward.
	 1. *subject_idx*, an integer speciying the index of the subject. This is for saving purposes. By default, it will be 0
	 1. *choices* , of shape (T) speciying whether the trials was a choice or a forced trial. By default, it will be np.ones(T), assuming thus there are no forced trials
 	1. *blocks_idx*, of shape (T), specifying the beginning of each blocks. If it is the beginning of a new block, a 1 should be present. By default, it will be set to blocks_idx = np.zeros(T), with blocks_idx[0]= 1, assuming thus only one block.
1. A *complete* argument in (1/0) specifying whether you are in the complete or feedback setting
1. A *leaky* argument in (-1/0/1) specifying the learning rule in the partial case. When complete, there is nothing to do, leaky=-1. In partial, one must choose between the leaky model or anticorrelated model leaky = 1/0; if leaky == 1, then the regression to the mean model will be applied; elif leaky == 0 then the 1 - R model will the applied else an error will be raise. Default is -1
1. An *onload* argument = True/False : If onload is set to True, the dictionary input expects much more variables (see load_results function). Essentially, it expects the contents of the output of the save function. Default is false
----
'''

info = {'actions':numpy.empty([1,1]), 'rewards':rewards_sim, 'subject_idx':sub_ind}

c=0 # complete
l=1 # leaky

'''
---- Infencence method ----
Takes as parameters:
	- noise = 1/0
	- apply_rep = 1/0
	- apply_weber = 1/0
	- condition = 1/0 : if noise = 1 , condition = observational_noise (noise in forced trials), if noise = 0, condition = apply_weber_decision_noise (weber-scaled softmax)
	- beta_softmax = -1/3 : softmax/argmax. If beta_softmax is set to 3, the value of the softmax parameter is 10**3 = 1000
	- temperature = temperature prior or beta prior. When inferring the beta, do we infer beta ~  U([0;100]) or T=1/beta ~ U([0;1]). 
					By default, we infer T=1/beta ~ U([0;1])
This function generates the posterior of the parameters as well as the marginal likelihood (model evidence) estimator
'''

# note: get traj not yet updated for task!!

'''
----noiseless RL model----
'''

#simulation
sim_obj = smc_object(info=info, complete = c, leaky = l)

sim_obj.map=param_sim

sim_obj.simulate(true_rewards=rewards_sim)
sim_obj.save_simulation(directory=RootDir + 'model_fit/results/noiselessRL/')

# recovery
sim_obj.actions=sim_obj.actions_simul
sim_obj.rewards=rewards_simul

sim_obj.do_inference(noise=0, apply_rep = 0, apply_weber = 0, condition=0, beta_softmax=-1, show_progress=show_prg) # exact RL
sim_obj.get_map()

sim_obj.save(directory=RootDir + 'model_fit/results/noiselessRL/')