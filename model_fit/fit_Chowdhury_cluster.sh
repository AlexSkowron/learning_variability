#!/bin/bash

ID="01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32"
drug_cond="placebo ldopa"

workdir="/home/mpib/skowron/aging_learning_var/learning_variability/model_fit"
logsdir="/home/mpib/skowron/aging_learning_var/learning_variability/model_fit/logs"

if [ ! -d ${logsdir} ]; then
	mkdir ${logsdir}
fi

for sub in ${ID}; do
	for drug in ${drug_cond}; do
		
		#G=$(ls "./results/simulation_Chowdhury/sub-${sub}_${drug}/noiselessRL_rep/" | wc -l)
		#
		#if [[ 12 != ${G} ]] ; then
			
  	  		sbatch -J fit_Chowdhury_sub-${sub}_${drug} -c 1 --time 12:0:0 --mem 1GB --workdir ${workdir} \
				--output ${logsdir}/slurm-%j.out --wrap "/home/mpib/skowron/.virtualenvs/RLvar/bin/python2 fit_Chowdhury.py $sub $drug"
		
			#fi
	done
done