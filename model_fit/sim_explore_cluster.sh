#!/bin/bash

workdir="/home/mpib/skowron/aging_learning_var/learning_variability/model_fit"
logsdir="/home/mpib/skowron/aging_learning_var/learning_variability/model_fit/sim_explore_logs"

model="noisyRL_softmax_rep"
#par="0" # 1-3 for noiselessRL noisyRL argmax; 1-4 for noisyRL softmax; 0 for all pars specified
#par_set="1 2 3 4 5 6 7 8 9 10 11" # 1-11 settings

if [ ! -d ${logsdir} ]; then
	mkdir ${logsdir}
fi

## par specific
#for sub in $(seq 1 30); do
#	for p in ${par}; do
#		for ps in ${par_set};do
#				sbatch -J sim_explore_${model}_${sub}_${p}_${ps} -c 1 --time 1:0:0 --mem 1GB --workdir ${workdir} \
#					--output ${logsdir}/slurm-%j.out --wrap "/home/mpib/skowron/.virtualenvs/RLvar/bin/python2 sim_${model}_explore.py $sub $p $ps"
#		done
#	done
#done

# sample all pars
for sub in $(seq 1 1000); do
	
	if [ ! -f ./results/sim_explore_all/${model}_alpha_U0_weber0/2q_complete0_subj${sub}_0_resInf1_map1_traj0_simul0_*_leaky1.mat ]; then
		
		
		sbatch -J sim_explore_all_${model}_${sub} -c 1 --time 1:0:0 --mem 1GB --workdir ${workdir} \
			--output ${logsdir}/slurm-%j.out --wrap "/home/mpib/skowron/.virtualenvs/RLvar/bin/python2 sim_${model}_explore.py $sub 0"
	fi

done