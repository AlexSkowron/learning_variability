% compute mutual information and proportion of non-greedy choices and save
% out

%% Mutual information across successive trials
clear all

load('/Volumes/fb-lip/Projects/UCL_Chowdhury_RL/data/behaviour/beh_data/beh_data.mat')

MI_subs_plac=cellfun(@mutual_info,A_placebo);
MI_subs_ldopa=cellfun(@mutual_info,A_ldopa);


%% Non-greedy choices
ID = {'01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32'};
drug_cond = {'placebo','ldopa'};

%prepare output
NonGreedy=zeros(length(ID),length(drug_cond));
highQ_RL=cell(length(ID),length(drug_cond));
NonGreedy_trials=cell(length(ID),length(drug_cond));

for i=1:length(ID)
    for d=1:length(drug_cond)
        
        if d==1
            a_sub=A_placebo{i};    
        elseif d==2
             a_sub=A_ldopa{i};  
        else
            fprintf('Error')
        end
  
    load(['/Users/skowron/Documents/Suboptimality_models/aging_learning_var/learning_variability/model_fit/results/fit_Chowdhury_RL_leaky1/2q_complete0_subj' ID{i} '_' drug_cond{d} '_resInf1_map1_traj1_simul0_param00001_leaky1.mat'],'results')
    
    [~,highQ]=max(results{2}'); % get greedy choices
    highQ_RL{i,d}=highQ;
    NonGreedyI=~(highQ'==(a_sub+1));
    NonGreedy_sub=sum(NonGreedyI)/length(a_sub); % fraction non-greedy choices
    
    NonGreedy_trials{i,d}=NonGreedyI;
    NonGreedy(i,d)=NonGreedy_sub;
    
    clear a_sub results highQ NonGreedy_sub NonGreedyI
    end
end

save('/Users/skowron/Documents/Suboptimality_models/aging_learning_var/learning_variability/model_comp/results/tradeoff_pers.mat','MI_subs_plac','MI_subs_ldopa','NonGreedy','highQ_RL','NonGreedy_trials')

%%
function MI = mutual_info(Ac)
% returns mutual information across successive choices

At=(Ac(2:end));
Att=(Ac(1:end-1)); % previous choice
Nt=length(Ac)-1;

jP=zeros(2,2); % joint probabilities; rows: Att, columns: At

jP(1,1)=sum(Att==0 & At==0)/Nt;
jP(2,1)=sum(Att==1 & At==0)/Nt;
jP(1,2)=sum(Att==0 & At==1)/Nt;
jP(2,2)=sum(Att==1 & At==1)/Nt;

% initialise output
MI=0;

for tt=1:2 %cycle over previous actions
    for t=1:2 % cycle over current actions
        
        if jP(tt,t)==0 % skips joint probabilites of zero to avoid nans. Ie. assumes MI for that event to be zero!
            continue
        end

        MI = MI + jP(tt,t)*log(jP(tt,t)/(sum(jP(tt,:))*sum(jP(:,t))));

    end
end
    
end

