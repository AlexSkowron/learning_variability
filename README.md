<h1> Noisy Inference </h1>

This is a project realised at Ecole Normale Supérieure under the supervision of Valentin Wyart and with collaborators Remi 
Dromnelle, Vasilisa Skvortsova and Charles Findling.

<h3> Summary of the paper </h3>

<h3> Summary of the code </h3>

This code offers flexible implementations in the case of Q-learning algorithms. Four big types of Q-learning algorithms are developped, obtained with combining noiseless and noisy algorithms with 1 or 2 Q-values. 

Multiple features are offered within the code. A first very important thing to understand when using this code is how to put data in the smc_object class. The constructor is expecting a dictionary with keys:
<ul>
  <li>'rewards' of shape (2, T) with T the length of the experiment. For partial cases, the unknown reward will be filled accordingly, (for examples, with zeros)</li>
  <li>'actions' of shape (T) speciying the actions of the subject</li>
  <li>'subject_idx', an integer speciying the index of the subject. This is for saving purposes. By default, it will be 0</li>
  <li>'choices', of shape (T) speciying whether the trials was a choice or a forced trial. By default, it will be np.ones(T), assuming thus there are no forced trials</li>
  <li>'blocks_idx', of shape (T), specifying the beginning of each blocks. If it is the beginning of a new block, a 1 should be present. By default, it will be set to idx_b = np.zeros(T), with idx_b[0]= 1, assuming thus only one block </li>
</ul>

<h3> Code compilation </h3>

To compile the c++ libraries, open the compile_c.sh file. Modify it by adding your boost library path. Then launch ./compile_c.sh

<h3> Code instructions </h3>

A simple example is given below. First, import the smc object file

```
from smc_object import smc_object  
```
Then, create an smc_object by specifying a path and the complete/partial argument
```
p = 'subj2_exp2.pkl'  
s_obj = smc_object(path = p, exp_idx = 2, complete = 1)
```
Do inference by specifying the wanted parameters. Then get MAP and smoothing trajectory
```
s_obj.do_inference(noise=1, apply_rep = 1, apply_weber = 1, apply_guided=1, beta_softmax=-1, show_progress=True)  
s_obj.get_map() 
s_obj.get_trajectory()  
```
Plot and save results
```
s_obj.plot()  
s_obj.save(directory='results/') 
```

 <h3> References </h3>


# learning_variability