#!/usr/bin/env python

from smc_object import smc_object
import pandas
import numpy

DataPath = '/Volumes/fb-lip/Projects/UCL_Chowdhury_RL/BIDS/data/'

ID = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32']

drug_cond = ['placebo','ldopa']

for sub in ID:
    for drug in drug_cond:

        '''
        load data
        '''
        
        p    = DataPath + 'sub-' + sub + '/ses-' + drug + '/func/sub-' + sub + '_task-bandit_run-all_events.tsv'
        sub_data = pandas.read_csv(p, sep="\t")
        
        sub_ind = sub + '_' + drug
        
        actions = numpy.array(sub_data.action)
        actions = actions[~numpy.isnan(actions)]
        actions = actions-1
        actions = actions.astype('uint8')
        
        rewards = numpy.array(sub_data.reward)
        rewards = rewards[~numpy.isnan(rewards)]
        
        rewards_reshape = numpy.zeros((2,len(rewards)))
        
        for i in range(1,len(rewards)): 
            rewards_reshape[actions[i],i]=rewards[i]
        
        rewards_reshape = rewards_reshape.astype('float') # must be float!
        
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
        
        info = {'actions':actions, 'rewards':rewards_reshape, 'subject_idx':sub_ind}
        
        s_obj = smc_object(info=info, complete = 0, leaky = 1)
        
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
        
        # noisy RL model
        s_obj.do_inference(noise=1, apply_rep = 0, apply_weber = 1, condition=1, beta_softmax=-1, show_progress=False)
        s_obj.get_map()
        
        s_obj.get_trajectory()
        
        s_obj.save(directory='/Users/skowron/Documents/Suboptimality_models/aging_learning_var/learning_variability/model_fit/results/fit_Chowdhury_RL/')
        
        # noisy RL model with perseveration
        s_obj.do_inference(noise=1, apply_rep = 1, apply_weber = 1, condition=1, beta_softmax=-1, show_progress=False)
        s_obj.get_map()
        
        s_obj.save(directory='/Users/skowron/Documents/Suboptimality_models/aging_learning_var/learning_variability/model_fit/results/fit_Chowdhury_RL/')
        
        # exact RL model with perseveration
        s_obj.do_inference(noise=0, apply_rep = 1, apply_weber = 0, condition=0, beta_softmax=-1, show_progress=True)
        s_obj.get_map() 
        
        s_obj.save(directory='/Users/skowron/Documents/Suboptimality_models/aging_learning_var/learning_variability/model_fit/results/fit_Chowdhury_RL/')
        
        # exact RL model
        s_obj.do_inference(noise=0, apply_rep = 0, apply_weber = 0, condition=0, beta_softmax=-1, show_progress=False)
        s_obj.get_map() 
        
        s_obj.save(directory='/Users/skowron/Documents/Suboptimality_models/aging_learning_var/learning_variability/model_fit/results/fit_Chowdhury_RL/')
        
        print('sub-' + sub + ' done')