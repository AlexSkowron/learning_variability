from smc_object import smc_object
import pandas
import numpy
import os
import sys
import pickle

sub=sys.argv[1]

print('starting ' + sub)

RootDir = '/home/mpib/skowron/aging_learning_var/learning_variability/'
PayOffDir = '/home/mpib/skowron/deBoer_RL/'
DataPath = '/home/mpib/skowron/deBoer_RL/beh_data/'

show_prg=False # show progress
simul=False # simulation on/off

PayOffScheme=numpy.genfromtxt(PayOffDir + 'PayOff.csv', delimiter=',')

if os.path.isdir(RootDir + 'model_fit/results/fit_deBoer_RL_weber0/') == 0:
    os.makedirs(RootDir + 'model_fit/results/fit_deBoer_RL_weber0/')

if simul:
    if os.path.isdir(RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/') == 0:
        os.makedirs(RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/')

'''
load data
'''
p    = DataPath + sub + '.tsv'
sub_data = pandas.read_csv(p, sep="\t")

sub_ind = sub

actions = numpy.array(sub_data.action) # no info which trials were nans
#actions = actions[~numpy.isnan(actions)]
actions = actions-1
actions = actions.astype('uint8')

rewards = numpy.array(sub_data.reward)
#rewards = rewards[~numpy.isnan(rewards)]

rewards_reshape = numpy.zeros((2,len(rewards)))

for i in range(1,len(rewards)): 
    rewards_reshape[actions[i],i]=rewards[i]

rewards_reshape = rewards_reshape.astype('float') # must be float!

reward_probs=numpy.genfromtxt(PayOffDir + 'PayOff.csv', delimiter=',') #experimental payoff scheme(s)
        
reward_probs = reward_probs.astype('float')

'''
---
1. A dictionary with keys
 	1. *actions*, of shape (T) speciying the actions of the subject - (0 or 1). T is the total number of trials
 	1. *rewards*, of shape (2, T) with T the length of the experiment. These should be normalised between 0 and 1. In the partial case, the learning rule of the unchosen option will override the unchosen reward.
	 1. *subject_idx*, an integer speciying the index of the subject. This is for saving purposes. By default, it will be 0
	 1. *choices* , of shape (T) speciying whether the trials was a choice or a forced trial. By default, it will be numpy.ones(T), assuming thus there are no forced trials
 	1. *blocks_idx*, of shape (T), specifying the beginning of each blocks. If it is the beginning of a new block, a 1 should be present. By default, it will be set to blocks_idx = numpy.zeros(T), with blocks_idx[0]= 1, assuming thus only one block.
1. A *complete* argument in (1/0) specifying whether you are in the complete or feedback setting
1. A *leaky* argument in (-1/0/1) specifying the learning rule in the partial case. When complete, there is nothing to do, leaky=-1. In partial, one must choose between the leaky model or anticorrelated model leaky = 1/0; if leaky == 1, then the regression to the mean model will be applied; elif leaky == 0 then the 1 - R model will the applied else an error will be raise. Default is -1
1. An *onload* argument = True/False : If onload is set to True, the dictionary input expects much more variables (see load_results function). Essentially, it expects the contents of the output of the save function. Default is false
----
'''

info = {'actions':actions, 'rewards':rewards_reshape, 'subject_idx':sub_ind}

c=0 # complete par
l=1 # leaky par. Note: leaky will not affect unchosen q value if alpha_unchosen is set to 0

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

# simulate rewards for recovery (simulation assumes choice for all 220 experimental trials!)
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

#note: get traj not yet updated for task!!

#'''
#----noiseless RL model----
#'''
#
#s_obj = smc_object(info=info, complete = c, leaky = l)
#        
#s_obj.do_inference(noise=0, apply_rep = 0, apply_weber = 0, condition=0, beta_softmax=-1, show_progress=show_prg)
#s_obj.get_map()
#
#beta_sim = s_obj.map[s_obj.param_names.index('beta_softmax')] # save beta for later simulation
#
#s_obj.save(directory=RootDir + 'model_fit/results/fit_deBoer_RL_weber0/')
#
#if simul:
#    
#    #simulation
#    if os.path.isdir(RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noiselessRL/') == 0:
#        os.makedirs(RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noiselessRL/')
#    
#    sim_obj = s_obj
#    
#    sim_obj.idx_blocks = numpy.zeros(rewards_sim.shape[1]); sim_obj.idx_blocks[0] = 1.
#    sim_obj.idx_blocks = numpy.asarray(sim_obj.idx_blocks, dtype=numpy.intc)
#    
#    sim_obj.choices = numpy.ones(rewards_sim.shape[1], dtype = numpy.intc)
#    
#    sim_obj.simulate(true_rewards=rewards_sim)
#    sim_obj.save_simulation(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noiselessRL/')
#    
#    #sim_info = pickle.load(open(RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noiselessRL/' + 'subj' + sub + '_simul1_2q_complete0_param00001_leaky1.pkl', "rb" ))
#    
#    # recovery
#    
#    #info_r = {'actions':sim_info["actions"], 'rewards':sim_info["rewards"], 'subject_idx':sub_ind + '_rec'}
#    info_r = {'actions':sim_obj.actions_simul, 'rewards':sim_obj.rewards_simul, 'subject_idx':sub_ind + '_rec'}
#    
#    r_obj = smc_object(info=info_r, complete = c, leaky = l)
#    
#    r_obj.do_inference(noise=0, apply_rep = 0, apply_weber = 0, condition=0, beta_softmax=-1, show_progress=show_prg) # exact RL
#    r_obj.get_map()
#    
#    r_obj.save(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noiselessRL/')
#    
#    r_obj.do_inference(noise=0, apply_rep = 1, apply_weber = 0, condition=0, beta_softmax=-1, show_progress=show_prg) # exact RL+rep
#    r_obj.get_map()
#    
#    r_obj.save(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noiselessRL/')
#    
#    r_obj.do_inference(noise=1, apply_rep = 0, apply_weber = 1, condition=1, beta_softmax=3, show_progress=show_prg) # noisy RL argmax
#    r_obj.get_map()
#    
#    r_obj.save(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noiselessRL/')
#    
#    r_obj.do_inference(noise=1, apply_rep = 0, apply_weber = 1, condition=1, beta_softmax=-1, show_progress=show_prg) # noisy RL softmax
#    r_obj.get_map()
#    
#    r_obj.save(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noiselessRL/')
#    
#    r_obj.do_inference(noise=1, apply_rep = 1, apply_weber = 1, condition=1, beta_softmax=-1, show_progress=show_prg) # noisy RL softmax + rep
#    r_obj.get_map()
#    
#    r_obj.save(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noiselessRL/')
#    
#    del sim_obj, r_obj, info_r
#
##del r_obj, info_r, sim_info
#del s_obj

'''
----noisy RL model with argmax----
'''

s_obj = smc_object(info=info, complete = c, leaky = l)

s_obj.do_inference(noise=1, apply_rep = 0, apply_weber = 0, condition=1, beta_softmax=3, show_progress=show_prg)
s_obj.get_map()

#s_obj.get_trajectory()

ep_sim = s_obj.map[s_obj.param_names.index('epsilon')] # save epsilon for later simulation

s_obj.save(directory=RootDir + 'model_fit/results/fit_deBoer_RL_weber0/')

if simul:
    
    #simulation
    if os.path.isdir(RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_argmax/') == 0:
        os.makedirs(RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_argmax/')
    
    sim_obj = s_obj
    
    sim_obj.idx_blocks = numpy.zeros(rewards_sim.shape[1]); sim_obj.idx_blocks[0] = 1.
    sim_obj.idx_blocks = numpy.asarray(sim_obj.idx_blocks, dtype=numpy.intc)
    
    sim_obj.choices = numpy.ones(rewards_sim.shape[1], dtype = numpy.intc)
    
    sim_obj.simulate(true_rewards=rewards_sim)
    sim_obj.save_simulation(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_argmax/')
    
    #sim_info = pickle.load(open(RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_argmax/' + 'subj' + sub + '_simul1_2q_complete0_param11010_leaky1.pkl', "rb" ))
    
    # recovery
    #info_r = {'actions':sim_info["actions"], 'rewards':sim_info["rewards"], 'subject_idx':sub_ind + '_rec'}
    info_r = {'actions':sim_obj.actions_simul, 'rewards':sim_obj.rewards_simul, 'subject_idx':sub_ind + '_rec'}
    
    r_obj = smc_object(info=info_r, complete = c, leaky = l)
    
    r_obj.do_inference(noise=0, apply_rep = 0, apply_weber = 0, condition=0, beta_softmax=-1, show_progress=show_prg) # exact RL
    r_obj.get_map()
    
    r_obj.save(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_argmax/')
    
    r_obj.do_inference(noise=0, apply_rep = 1, apply_weber = 0, condition=0, beta_softmax=-1, show_progress=show_prg) # exact RL + rep
    r_obj.get_map()
    
    r_obj.save(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_argmax/')
    
    r_obj.do_inference(noise=1, apply_rep = 0, apply_weber = 1, condition=1, beta_softmax=3, show_progress=show_prg) # noisy RL argmax
    r_obj.get_map()
    
    r_obj.save(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_argmax/')
    
    r_obj.do_inference(noise=1, apply_rep = 0, apply_weber = 1, condition=1, beta_softmax=-1, show_progress=show_prg) # noisy RL softmax
    r_obj.get_map()
    
    r_obj.save(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_argmax/')
    
    r_obj.do_inference(noise=1, apply_rep = 1, apply_weber = 1, condition=1, beta_softmax=-1, show_progress=show_prg) # noisy RL softmax + rep
    r_obj.get_map()
    
    r_obj.save(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_argmax/')
    
    del sim_obj, r_obj, info_r

#del r_obj, info_r, sim_info
del s_obj

'''
-----noisy RL model with softmax-----
'''

s_obj = smc_object(info=info, complete = c, leaky = l)

s_obj.do_inference(noise=1, apply_rep = 0, apply_weber = 0, condition=1, beta_softmax=-1, show_progress=show_prg)
s_obj.get_map()

#s_obj.get_trajectory()

s_obj.save(directory=RootDir + 'model_fit/results/fit_deBoer_RL_weber0/')

if simul:
    
    #simulation
    if os.path.isdir(RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_softmax/') == 0:
        os.makedirs(RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_softmax/')
    
    sim_obj = s_obj
    
    sim_obj.idx_blocks = numpy.zeros(rewards_sim.shape[1]); sim_obj.idx_blocks[0] = 1.
    sim_obj.idx_blocks = numpy.asarray(sim_obj.idx_blocks, dtype=numpy.intc)
    
    sim_obj.choices = numpy.ones(rewards_sim.shape[1], dtype = numpy.intc)
    
    #simulate noisyRL softmax with beta map from exact model and epsilon map from noisy RL argmax
    sim_obj.map[s_obj.param_names.index('beta_softmax')]=beta_sim
    sim_obj.map[s_obj.param_names.index('epsilon')]=ep_sim
    
    sim_obj.simulate(true_rewards=rewards_sim)
    sim_obj.save_simulation(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_softmax/')
    
    #sim_info = pickle.load(open(RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_softmax/' + 'subj' + sub + '_simul1_2q_complete0_param11011_leaky1.pkl', "rb" ))
    
    ## recovery
    #info_r = {'actions':sim_info["actions"], 'rewards':sim_info["rewards"], 'subject_idx':sub_ind + '_rec'}
    info_r = {'actions':sim_obj.actions_simul, 'rewards':sim_obj.rewards_simul, 'subject_idx':sub_ind + '_rec'}
    
    r_obj = smc_object(info=info_r, complete = c, leaky = l)
    
    r_obj.do_inference(noise=0, apply_rep = 0, apply_weber = 0, condition=0, beta_softmax=-1, show_progress=show_prg) # exact RL
    r_obj.get_map()
    
    r_obj.save(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_softmax/')
    
    r_obj.do_inference(noise=0, apply_rep = 1, apply_weber = 0, condition=0, beta_softmax=-1, show_progress=show_prg) # exact RL + rep
    r_obj.get_map()
    
    r_obj.save(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_softmax/')
    
    r_obj.do_inference(noise=1, apply_rep = 0, apply_weber = 1, condition=1, beta_softmax=3, show_progress=show_prg) # noisy RL argmax
    r_obj.get_map()
    
    r_obj.save(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_softmax/')
    
    r_obj.do_inference(noise=1, apply_rep = 0, apply_weber = 1, condition=1, beta_softmax=-1, show_progress=show_prg) # noisy RL softmax
    r_obj.get_map()
    
    r_obj.save(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_softmax/')
    
    r_obj.do_inference(noise=1, apply_rep = 1, apply_weber = 1, condition=1, beta_softmax=-1, show_progress=show_prg) # noisy RL softmax + rep
    r_obj.get_map()
    
    r_obj.save(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_softmax/')
    
    del sim_obj, r_obj, info_r

#del r_obj, info_r, sim_info
del s_obj

'''
noisy RL model with perseveration
'''
s_obj = smc_object(info=info, complete = c, leaky = l)

s_obj.do_inference(noise=1, apply_rep = 1, apply_weber = 0, condition=1, beta_softmax=-1, show_progress=show_prg)
s_obj.get_map()

s_obj.save(directory=RootDir + 'model_fit/results/fit_deBoer_RL_weber0/')

if simul:
    
    #simulation
    if os.path.isdir(RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_softmax_rep/') == 0:
        os.makedirs(RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_softmax_rep/')
    
    sim_obj = s_obj
    
    sim_obj.idx_blocks = numpy.zeros(rewards_sim.shape[1]); sim_obj.idx_blocks[0] = 1.
    sim_obj.idx_blocks = numpy.asarray(sim_obj.idx_blocks, dtype=numpy.intc)
    
    sim_obj.choices = numpy.ones(rewards_sim.shape[1], dtype = numpy.intc)
    
    sim_obj.simulate(true_rewards=rewards_sim)
    sim_obj.save_simulation(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_softmax_rep/')
    
    #sim_info = pickle.load(open(RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_softmax_rep/' + 'subj' + sub + '_simul1_2q_complete0_param11111_leaky1.pkl', "rb" ))
    
    ##recovery
    
    #info_r = {'actions':sim_info["actions"], 'rewards':sim_info["rewards"], 'subject_idx':sub_ind + '_rec'}
    info_r = {'actions':sim_obj.actions_simul, 'rewards':sim_obj.rewards_simul, 'subject_idx':sub_ind + '_rec'}
    
    r_obj = smc_object(info=info_r, complete = c, leaky = l)
    
    r_obj.do_inference(noise=0, apply_rep = 0, apply_weber = 0, condition=0, beta_softmax=-1, show_progress=show_prg) # exact RL
    r_obj.get_map()
    
    r_obj.save(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_softmax_rep/')
    
    r_obj.do_inference(noise=0, apply_rep = 1, apply_weber = 0, condition=0, beta_softmax=-1, show_progress=show_prg) # exact RL + rep
    r_obj.get_map()
    
    r_obj.save(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_softmax_rep/')
    
    r_obj.do_inference(noise=1, apply_rep = 0, apply_weber = 1, condition=1, beta_softmax=3, show_progress=show_prg) # noisy RL argmax
    r_obj.get_map()
    
    r_obj.save(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_softmax_rep/')
    
    r_obj.do_inference(noise=1, apply_rep = 0, apply_weber = 1, condition=1, beta_softmax=-1, show_progress=show_prg) # noisy RL softmax
    r_obj.get_map()
    
    r_obj.save(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_softmax_rep/')
    
    r_obj.do_inference(noise=1, apply_rep = 1, apply_weber = 1, condition=1, beta_softmax=-1, show_progress=show_prg) # noisy RL softmax + rep
    r_obj.get_map()
    
    r_obj.save(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_softmax/')
    
    del sim_obj, r_obj, info_r

#del r_obj, info_r, sim_info
del s_obj

#'''
#exact RL model with perseveration
#'''
#
#s_obj = smc_object(info=info, complete = c, leaky = l)
#
#s_obj.do_inference(noise=0, apply_rep = 1, apply_weber = 0, condition=0, beta_softmax=-1, show_progress=show_prg)
#s_obj.get_map() 
#
#s_obj.save(directory=RootDir + 'model_fit/results/fit_deBoer_RL_weber0/')
#
#if simul:
#    
#    #simulation
#    if os.path.isdir(RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noiselessRL_rep/') == 0:
#        os.makedirs(RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noiselessRL_rep/')
#    
#    sim_obj = s_obj
#    
#    sim_obj.idx_blocks = numpy.zeros(rewards_sim.shape[1]); sim_obj.idx_blocks[0] = 1.
#    sim_obj.idx_blocks = numpy.asarray(sim_obj.idx_blocks, dtype=numpy.intc)
#    
#    sim_obj.choices = numpy.ones(rewards_sim.shape[1], dtype = numpy.intc)
#    
#    sim_obj.simulate(true_rewards=rewards_sim)
#    sim_obj.save_simulation(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noiselessRL_rep/')
#    
#    #sim_info = pickle.load(open(RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noiselessRL_rep/' + 'subj' + sub + '_simul1_2q_complete0_param00101_leaky1.pkl', "rb" ))
#    
#    #recovery
#    
#    #info_r = {'actions':sim_info["actions"], 'rewards':sim_info["rewards"], 'subject_idx':sub_ind + '_rec'}
#    info_r = {'actions':sim_obj.actions_simul, 'rewards':sim_obj.rewards_simul, 'subject_idx':sub_ind + '_rec'}
#    
#    r_obj = smc_object(info=info_r, complete = c, leaky = l)
#    
#    r_obj.do_inference(noise=0, apply_rep = 0, apply_weber = 0, condition=0, beta_softmax=-1, show_progress=show_prg) # exact RL
#    r_obj.get_map()
#    
#    r_obj.save(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noiselessRL_rep/')
#    
#    r_obj.do_inference(noise=0, apply_rep = 1, apply_weber = 0, condition=0, beta_softmax=-1, show_progress=show_prg) # exact RL + rep
#    r_obj.get_map()
#    
#    r_obj.save(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noisyRL_softmax_rep/')
#    
#    r_obj.do_inference(noise=1, apply_rep = 0, apply_weber = 1, condition=1, beta_softmax=3, show_progress=show_prg) # noisy RL argmax
#    r_obj.get_map()
#    
#    r_obj.save(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noiselessRL_rep/')
#    
#    r_obj.do_inference(noise=1, apply_rep = 0, apply_weber = 1, condition=1, beta_softmax=-1, show_progress=show_prg) # noisy RL softmax
#    r_obj.get_map()
#    
#    r_obj.save(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noiselessRL_rep/')
#    
#    r_obj.do_inference(noise=1, apply_rep = 1, apply_weber = 1, condition=1, beta_softmax=-1, show_progress=show_prg) # noisy RL softmax + rep
#    r_obj.get_map()
#    
#    r_obj.save(directory=RootDir + 'model_fit/results/simulation_deBoer_weber0/' + sub + '/noiselessRL_rep/')
#    
#    del sim_obj, r_obj, info_r
#
##del r_obj, info_r, sim_info
#del s_obj

print(sub + ' done!')