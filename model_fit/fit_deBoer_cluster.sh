#!/bin/bash

ID="dad02_OA dad05_OA dad13_OA dad16_OA dad19_OA dad22_OA dad25_OA dad30_OA dad34_YA dad37_YA dad40_YA dad46_YA dad49_YA dad55_YA dad61_OA dad64_YA dad67_YA dad80_YA dad84_OA dad90_YA dad03_OA dad08_OA dad14_OA dad17_OA dad20_OA dad23_OA dad26_OA dad31_OA dad35_YA dad38_YA dad42_YA dad47_YA dad50_YA dad58_YA dad62_YA dad65_YA dad70_YA dad82_YA dad85_OA dad04_OA dad11_OA dad15_OA dad18_OA dad21_OA dad24_OA dad29_OA dad33_OA dad36_YA dad39_YA dad43_YA dad48_YA dad52_YA dad60_OA dad63_YA dad66_YA dad72_YA dad83_YA dad86_YA"

#dad11_OA, dad31_OA and dad26_OA excluded due to lack of behavioural variability

workdir="/home/mpib/skowron/aging_learning_var/learning_variability/model_fit"
logsdir="/home/mpib/skowron/aging_learning_var/learning_variability/model_fit/logs"

if [ ! -d ${logsdir} ]; then
	mkdir ${logsdir}
fi

for sub in ${ID}; do
	sbatch -J fit_deBoer_${sub} -c 1 --time 12:0:0 --mem 1GB --workdir ${workdir} \
		--output ${logsdir}/slurm-%j.out --wrap "/home/mpib/skowron/.virtualenvs/RLvar/bin/python2 fit_deBoer.py $sub"
done