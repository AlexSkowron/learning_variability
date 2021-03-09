% model recovery
clear all

addpath(genpath('/Users/skowron/Documents/MATLAB/VBA-toolbox-master'))

load('/Users/skowron/Volumes/tardis/skowron/deBoer_RL/TAB_for_Alex2.mat','subjects','group')

ID_OA = cellfun(@(x) [x '_OA'],subjects(group==2),'UniformOutput',false);
ID_YA = cellfun(@(x) [x '_YA'],subjects(group==1),'UniformOutput',false);

%remove excluded OA subjects
ID_OA(find(strcmp(ID_OA,'dad11_OA')))=[];
ID_OA(find(strcmp(ID_OA,'dad31_OA')))=[];
ID_OA(find(strcmp(ID_OA,'dad26_OA')))=[];

model_ID = {'param00001','param00101','param11010','param11011','param11111'}; % must map to model name
model_name = {'noiselessRL' 'noiselessRL_rep' 'noisyRL_argmax' 'noisyRL_softmax' 'noisyRL_softmax_rep'};

SimPath='/Users/skowron/Volumes/tardis/skowron/aging_learning_var/learning_variability/model_fit/results/simulation_deBoer/';

for m = 1:length(model_name)
    
    model_name{m}
    
    for age_gr = 1:2 % perform model comparison for each age group

        if age_gr==1
            ID=ID_YA;
            gr_label='YA';
        elseif age_gr==2
            ID=ID_OA;
            gr_label='OA';
        else
            assert(0,'no age group specified')
        end

        % retrieve data
        Evidences=zeros(length(model_ID),length(ID));

        for sub = 1:length(ID)    
            for model = 1:length(model_ID)

                load([SimPath ID{sub} '/' model_name{m} '/2q_complete0_subj' ID{sub} '_rec_resInf1_map1_traj0_simul0_' model_ID{model} '_leaky1.mat'],'results')

                Evidences(model,sub)=results{end}(end);

                clear results

            end
        end

        % check model recovery
        gr_label
        [posterior,out] = VBA_groupBMC(Evidences(:,:,1));

        pause
    
    end
end