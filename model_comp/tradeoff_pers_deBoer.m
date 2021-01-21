% compute mutual information and proportion of non-greedy choices and save
% out
clear all

load('/Volumes/fb-lip/user/Alexander/Lieke_bandit_data/TAB_for_Alex.mat','subjects','group','A')
% 0 1 coding for actions
A=cellfun(@(x) x-1,A,'UniformOutput',false);

ID_OA = cellfun(@(x) [x '_OA'],subjects(group==2),'UniformOutput',false);
ID_YA = cellfun(@(x) [x '_YA'],subjects(group==1),'UniformOutput',false);

age_gr={'YA' 'OA'};

for a = 1:length(age_gr) %cycle over age groups
    
    if a==1
        ID=ID_YA;
    elseif a==2
        ID=ID_OA;
    end
    
    %prepare output
    NonGreedy=zeros(length(ID),1);
    highQ_RL=cell(length(ID),1);
    NonGreedy_trials=cell(length(ID),1);
    
    %compute MI
    MI_subs=cellfun(@mutual_info,A(group==a));
    
    %compute behavioural correlations
    BEH_correl_subs=cellfun(@beh_corr,A(group==a));
    
    for i=1:length(ID)

        load(['/Users/skowron/Documents/Suboptimality_models/aging_learning_var/learning_variability/model_fit/results/fit_deBoer_RL_leaky1/2q_complete0_subj' ID{i} '_resInf1_map1_traj1_simul0_param00001_leaky1.mat'],'results','actions')

        [~,highQ]=max(results{2}'); % get greedy choices
        highQ_RL{i}=highQ;
        NonGreedyI=~(highQ==(actions+1));
        NonGreedy_sub=sum(NonGreedyI)/length(actions); % fraction non-greedy choices

        NonGreedy_trials{i}=NonGreedyI;
        NonGreedy(i)=NonGreedy_sub;

        clear actions results highQ NonGreedy_sub NonGreedyI
    end

    save(['/Users/skowron/Documents/Suboptimality_models/aging_learning_var/learning_variability/model_comp/results/tradeoff_pers_deBoer_' age_gr{a} '.mat'],'MI_subs','BEH_correl_subs','NonGreedy','highQ_RL','NonGreedy_trials')

end

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

function BC = beh_corr(Ac)
% compute behavioural correlation of successive actions

At=(Ac(2:end));
Att=(Ac(1:end-1)); % previous choice

BC=corr(At,Att);

end
