
############################################################################################
#  		      	Compilation file				         
#											                       
############################################################################################

# path to boost library : TO BE MODIFIED BY USER
path='/home/mpib/skowron/aging_learning_var/libraries/boost_1_59_0'

# compilation 
cd lib_c/smc2/
rm smc_py.cpp
python setup.py build_ext --inplace --include-dirs=$path
cd ../state_estimation/
rm smc_py.cpp
python setup.py build_ext --inplace --include-dirs=$path 