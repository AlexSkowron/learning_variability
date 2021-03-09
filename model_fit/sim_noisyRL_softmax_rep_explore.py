from smc_object import smc_object
import numpy
import os
import sys

sub=sys.argv[1]
par=int(sys.argv[2])

if par > 0:
    par_set=int(sys.argv[3])
    
    if par == 1: # alpha chosen
        par_simul=numpy.arange(start=0, stop=1.1, step=0.1)
        param_sim=numpy.array([par_simul[par_set-1], 0.3, 20, 0.2, 2])
    elif par == 2: # alpha unchosen
        par_simul=numpy.arange(start=0, stop=1.1, step=0.1)
        param_sim=numpy.array([0.8, par_simul[par_set-1], 20, 0.2, 2])
    elif par == 3: # beta
        par_simul=numpy.arange(start=0.1, stop=110.1, step=10)
        param_sim=numpy.array([0.8, 0.3, par_simul[par_set-1], 0.2, 2])
    elif par == 4: # epsilon
        par_simul=numpy.arange(start=0, stop=1.1, step=0.1)
        param_sim=numpy.array([0.8, 0.3, 20, par_simul[par_set-1], 2])
    elif par == 5: # eta
        par_simul=numpy.arange(start=-10, stop=10.1, step=2)
        param_sim=numpy.array([0.8, 0.3, 20, 0.2, par_simul[par_set-1]])
    else:
        raise ValueError('No simulation was done')
    
    OutDir = '/home/mpib/skowron/aging_learning_var/learning_variability/model_fit/results/sim_explore/noisyRL_softmax_rep_alpha_U0_weber0/'
    sub_ind = sub + '_' + str(par) + '_' + str(par_set)
    
    print('starting sim_sub-' + sub + '_par' + str(par) + '_ite' + str(par_set))
    
elif par == 0:
    param_sim=numpy.array([numpy.random.random_sample(size=None), 0, 50 * numpy.random.random_sample(size=None), numpy.random.random_sample(size=None), 20 * numpy.random.random_sample(size=None) - 10])
    #param_sim=numpy.array([numpy.random.random_sample(size=None), numpy.random.random_sample(size=None), 50 * numpy.random.random_sample(size=None), numpy.random.random_sample(size=None), 20 * numpy.random.random_sample(size=None) - 10]) 
    # ranges: alpha chosen [0,1), alpha unchosen [0,1), beta [0,50), epsilon [0,1), eta [-10,10)
    
    OutDir = '/home/mpib/skowron/aging_learning_var/learning_variability/model_fit/results/sim_explore_all/noisyRL_softmax_rep_alpha_U0_weber0/'
    sub_ind = sub + '_' + str(par)
    
    print('starting sim_sub-' + sub + '_par' + str(par))
    
else:
    raise ValueError('Specify simulation parameter')
    
PayOffDir = '/home/mpib/skowron/deBoer_RL/'

show_prg=False # show progress

reward_probs=numpy.genfromtxt(PayOffDir + 'PayOff.csv', delimiter=',') #experimental payoff scheme    
reward_probs = reward_probs.astype('float')

if os.path.isdir(OutDir) == 0:
    os.makedirs(OutDir)

'''
load data
'''

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
sim_info = {'actions':rewards_sim[1,:], 'rewards':rewards_sim, 'subject_idx':sub_ind} #placeholder actions

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

'''
----noiseless RL model----
'''

#simulation
sim_obj=smc_object(info=sim_info, complete = c, leaky = l)

sim_obj.traj_param={'noise': 1, 'apply_rep': 1, 'apply_weber': 0, 'beta_softmax': -1, 'condition': 1}
sim_obj.map=param_sim
sim_obj.got_map=1
sim_obj.simulate(true_rewards=rewards_sim)
sim_obj.save_simulation(directory=OutDir)

# recovery
r_info = {'actions':sim_obj.actions_simul, 'rewards':sim_obj.rewards_simul, 'subject_idx':sub_ind}

r_obj=smc_object(info=r_info, complete = c, leaky = l)

# noisyRL softmax
r_obj.do_inference(noise=1, apply_rep = 1, apply_weber = 0, condition=1, beta_softmax=-1, alpha_unchosen=0, show_progress=show_prg)
r_obj.get_map()

r_obj.save(directory=OutDir)