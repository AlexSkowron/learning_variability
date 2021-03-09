% model recovery
clear all

addpath(genpath('/Users/skowron/Documents/MATLAB/VBA-toolbox-master'))

ID = {'01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32'};

drug_cond = {'placebo','ldopa'};

model_ID = {'param00001','param00101','param11010','param11011','param11111'}; % must map to model name
model_name = {'noiselessRL' 'noiselessRL_rep' 'noisyRL_argmax' 'noisyRL_softmax' 'noisyRL_softmax_rep'};

SimPath='/Users/skowron/Volumes/tardis/skowron/aging_learning_var/learning_variability/model_fit/results/simulation_Chowdhury/';

for m = 3:length(model_name)
    
    model_name{m}
    
    % retrieve data
    Evidences=zeros(length(model_ID),length(ID),length(drug_cond));
    
    for sub = 1:length(ID)
       for drug = 1:length(drug_cond)      
            for model = 1:length(model_ID)

                load([SimPath 'sub-' ID{sub} '_' drug_cond{drug} '/' model_name{m} '/2q_complete0_subj' ID{sub} '_' drug_cond{drug} '_rec_resInf1_map1_traj0_simul0_' model_ID{model} '_leaky1.mat'],'results')

                Evidences(model,sub,drug)=results{end}(end);
                clear results

            end

       end
    end
    
    % check model recovery
    %placebo sim
    fprintf('placebo\n')
    [posterior,out] = VBA_groupBMC(Evidences(:,:,1));
    
    pause
    
    %ldopa sim
    fprintf('ldopa\n')
    [posterior,out] = VBA_groupBMC(Evidences(:,:,2));
    
    pause
    
    %collapsed sim
    fprintf('collapsed\n')
    [posterior,out] = VBA_groupBMC([Evidences(:,:,1) Evidences(:,:,2)]);
    
    pause
    
end