% Model comp across datasets
clear all

addpath(genpath('/Users/skowron/Documents/MATLAB/VBA-toolbox-master'))

cd /Users/skowron/Documents/Suboptimality_models/learning_variability

% Chowdhury et al data
RL1_dat=load('/Users/skowron/Documents/Suboptimality_models/aging_learning_var/learning_variability/model_fit/results/fit_Chowdhury_RL_leaky1/Evidences+pars.mat');

% deBoer et al data
RL2_dat.OA=load('/Users/skowron/Documents/Suboptimality_models/aging_learning_var/learning_variability/model_fit/results/fit_deBoer_RL_leaky1/Evidences+pars_OA.mat');
RL2_dat.YA=load('/Users/skowron/Documents/Suboptimality_models/aging_learning_var/learning_variability/model_fit/results/fit_deBoer_RL_leaky1/Evidences+pars_YA.mat');

%log transform beta parameters
load('/Users/skowron/Documents/Suboptimality_models/aging_learning_var/learning_variability/model_fit/results/fit_deBoer_RL_leaky1/2q_complete0_subjdad02_OA_resInf1_map1_traj1_simul0_param11111_leaky1.mat','param_names') % load param names

% log transform beta!!
RL1_dat.par_RL_noisy_rep(:,3,:)=log(RL1_dat.par_RL_noisy_rep(:,3,:));
RL2_dat.OA.par_RL_noisy_rep(:,3)=log(RL2_dat.OA.par_RL_noisy_rep(:,3));

param_names(3,:)='log_beta      ';


%% first check if different datasets differ in model fit

[h,p]=VBA_groupBMC_btwGroups({RL1_dat.Evidences(:,:,1),RL2_dat.OA.Evidences}) % no differences between OAs

[h,p]=VBA_groupBMC_btwGroups({[RL1_dat.Evidences(:,:,1) RL2_dat.OA.Evidences],RL2_dat.YA.Evidences}) % also no differences between YA and OA

%% analysis for OA collapsed across datasets (baseline/placebo)

% correlate parameters to task performance

% total rewards earned
load('/Volumes/fb-lip/Projects/UCL_Chowdhury_RL/data/behaviour/task_performance/reward.mat')
RL1_reward_OA=out_tab.money_won_placebo.*10; % convert from money won to rewarded trials

load('/Volumes/fb-lip/user/Alexander/Lieke_bandit_data/performance.mat','total_reward_OA')
RL2_reward_OA=total_reward_OA;

for l = 1:size(param_names,1)

    plot([RL1_dat.par_RL_noisy_rep(RL1_dat.sam_ind_RFX,l,1);RL2_dat.OA.par_RL_noisy_rep(RL2_dat.OA.sam_ind_RFX,l)],[RL1_reward_OA(RL1_dat.sam_ind_RFX);RL2_reward_OA(RL2_dat.OA.sam_ind_RFX)'],'bo')
    lsline

    [r,p]=corr([RL1_dat.par_RL_noisy_rep(RL1_dat.sam_ind_RFX,l,1);RL2_dat.OA.par_RL_noisy_rep(RL2_dat.OA.sam_ind_RFX,l)],[RL1_reward_OA(RL1_dat.sam_ind_RFX);RL2_reward_OA(RL2_dat.OA.sam_ind_RFX)']);

    title(['noisyRL+rep OA, r=' num2str(r) ', p=' num2str(p)])
    xlabel(param_names(l,:))
    ylabel('total reward')

    pause
    
end

%switches
load('/Volumes/fb-lip/Projects/UCL_Chowdhury_RL/data/behaviour/task_performance/switches.mat','switches_plac_prop')
RL1_switches_OA=switches_plac_prop;

load('/Volumes/fb-lip/user/Alexander/Lieke_bandit_data/performance.mat','switches_prop_OA')
RL2_switches_OA=switches_prop_OA;

for l = 1:size(param_names,1)

    plot([RL1_dat.par_RL_noisy_rep(RL1_dat.sam_ind_RFX,l,1);RL2_dat.OA.par_RL_noisy_rep(RL2_dat.OA.sam_ind_RFX,l)],[RL1_switches_OA(RL1_dat.sam_ind_RFX);RL2_switches_OA(RL2_dat.OA.sam_ind_RFX)'],'bo')
    lsline

    [r,p]=corr([RL1_dat.par_RL_noisy_rep(RL1_dat.sam_ind_RFX,l,1);RL2_dat.OA.par_RL_noisy_rep(RL2_dat.OA.sam_ind_RFX,l)],[RL1_switches_OA(RL1_dat.sam_ind_RFX);RL2_switches_OA(RL2_dat.OA.sam_ind_RFX)']);

    title(['noisyRL+rep OA, r=' num2str(r) ', p=' num2str(p)])
    xlabel(param_names(l,:))
    ylabel('Switches prop')

    pause
    
end

%objectively adaptive switches
load('/Volumes/fb-lip/Projects/UCL_Chowdhury_RL/data/behaviour/task_performance/Obj_adapt_switches.mat','obj_adapt_switches_perc_placebo')
RL1_obj_switches_OA=obj_adapt_switches_perc_placebo;

load('/Volumes/fb-lip/user/Alexander/Lieke_bandit_data/performance.mat','obj_adapt_switches_prop_OA')
RL2_obj_switches_OA=obj_adapt_switches_prop_OA;

for l = 1:size(param_names,1)

    plot([RL1_dat.par_RL_noisy_rep(RL1_dat.sam_ind_RFX,l,1);RL2_dat.OA.par_RL_noisy_rep(RL2_dat.OA.sam_ind_RFX,l)],[RL1_obj_switches_OA(RL1_dat.sam_ind_RFX);RL2_obj_switches_OA(RL2_dat.OA.sam_ind_RFX)'],'bo')
    lsline

    [r,p]=corr([RL1_dat.par_RL_noisy_rep(RL1_dat.sam_ind_RFX,l,1);RL2_dat.OA.par_RL_noisy_rep(RL2_dat.OA.sam_ind_RFX,l)],[RL1_obj_switches_OA(RL1_dat.sam_ind_RFX);RL2_obj_switches_OA(RL2_dat.OA.sam_ind_RFX)']);

    title(['noisyRL+rep OA, r=' num2str(r) ', p=' num2str(p)])
    xlabel(param_names(l,:))
    ylabel('Obj switches prop')

    pause
    
end

% correlate within-subject fitted rep bias in RL+rep model to epsilon in
% noisyRL model

%OA
plot([RL1_dat.par_RL_noisy(RL1_dat.sam_ind_RFX,4,1);RL2_dat.OA.par_RL_noisy(RL2_dat.OA.sam_ind_RFX,4)],[RL1_dat.par_RL_rep(RL1_dat.sam_ind_RFX,4,1);RL2_dat.OA.par_RL_rep(RL2_dat.OA.sam_ind_RFX,4)],'o')
lsline
xlabel('epsilon')
ylabel('rep_bias')

[r,p]=corr([RL1_dat.par_RL_noisy(RL1_dat.sam_ind_RFX,4,1);RL2_dat.OA.par_RL_noisy(RL2_dat.OA.sam_ind_RFX,4)],[RL1_dat.par_RL_rep(RL1_dat.sam_ind_RFX,4,1);RL2_dat.OA.par_RL_rep(RL2_dat.OA.sam_ind_RFX,4)])
title(['OA r=' num2str(r) ', p=' num2str(p)])

pause

%% plot trade-off between fraction non-greedy/switches and MI across successive trials by model log evidence difference between noisyRL and RL+rep
load('/Volumes/fb-lip/Projects/Naftali/data/analysis/PLS/figures/Z_archive/pre_commonCoord_correction/custom_colorbars/CustomColors.mat','hot_cool_c')

load('/Users/skowron/Documents/Suboptimality_models/learning_variability-master/tradeoff_pers.mat','MI_subs_plac','NonGreedy')
RL1_MI=MI_subs_plac;
RL1_NonGreedy=NonGreedy;
clear MI_subs_plac NonGreedy

load('/Users/skowron/Documents/Suboptimality_models/learning_variability-master/tradeoff_pers_deBoer_OA.mat','MI_subs','NonGreedy')
RL2_MI=MI_subs;
RL2_NonGreedy=NonGreedy;
clear MI_subs_plac NonGreedy

RL1_Ediff_OA=RL1_dat.Evidences(3,RL1_dat.sam_ind_RFX,1)-RL1_dat.Evidences(2,RL1_dat.sam_ind_RFX,1); % pos in favor of noisyRL
RL2_Ediff_OA=RL2_dat.OA.Evidences(3,RL2_dat.OA.sam_ind_RFX)-RL2_dat.OA.Evidences(2,RL2_dat.OA.sam_ind_RFX);

scatter([RL1_NonGreedy(RL1_dat.sam_ind_RFX)';RL2_NonGreedy(RL2_dat.OA.sam_ind_RFX)], [RL1_MI(RL1_dat.sam_ind_RFX);RL2_MI(RL2_dat.OA.sam_ind_RFX)'], 3000, [RL1_Ediff_OA';RL2_Ediff_OA'],'.');
colorbar
caxis([min(RL1_Ediff_OA) -min(RL1_Ediff_OA)])
colormap(hot_cool_c)
title('OA Model log evidence diff RLnoisy - RL+rep')
xlabel('Fraction non-greedy')
ylabel('Mutual information')
set(gca,'FontSize',30)

pause

% for noisyRL-RL

RL1_Ediff_OA=RL1_dat.Evidences(3,RL1_dat.sam_ind_RFX,1)-RL1_dat.Evidences(1,RL1_dat.sam_ind_RFX,1); % pos in favor of noisyRL
RL2_Ediff_OA=RL2_dat.OA.Evidences(3,RL2_dat.OA.sam_ind_RFX)-RL2_dat.OA.Evidences(1,RL2_dat.OA.sam_ind_RFX);

scatter([RL1_NonGreedy(RL1_dat.sam_ind_RFX)';RL2_NonGreedy(RL2_dat.OA.sam_ind_RFX)], [RL1_MI(RL1_dat.sam_ind_RFX);RL2_MI(RL2_dat.OA.sam_ind_RFX)'], 3000, [RL1_Ediff_OA';RL2_Ediff_OA'],'.');
colorbar
caxis([-max(RL2_Ediff_OA) max(RL2_Ediff_OA)])
colormap(hot_cool_c)
title('OA Model log evidence diff RLnoisy - RL')
xlabel('Fraction non-greedy')
ylabel('Mutual information')
set(gca,'FontSize',30)

pause

% for RL+rep-RL

RL1_Ediff_OA=RL1_dat.Evidences(2,RL1_dat.sam_ind_RFX,1)-RL1_dat.Evidences(1,RL1_dat.sam_ind_RFX,1); % pos in favor of noisyRL
RL2_Ediff_OA=RL2_dat.OA.Evidences(2,RL2_dat.OA.sam_ind_RFX)-RL2_dat.OA.Evidences(1,RL2_dat.OA.sam_ind_RFX);

scatter([RL1_NonGreedy(RL1_dat.sam_ind_RFX)';RL2_NonGreedy(RL2_dat.OA.sam_ind_RFX)], [RL1_MI(RL1_dat.sam_ind_RFX);RL2_MI(RL2_dat.OA.sam_ind_RFX)'], 3000, [RL1_Ediff_OA';RL2_Ediff_OA'],'.');
colorbar
caxis([-max(RL2_Ediff_OA) max(RL2_Ediff_OA)])
colormap(hot_cool_c)
title('OA Model log evidence diff RL+rep - RL')
xlabel('Fraction non-greedy')
ylabel('Mutual information')
set(gca,'FontSize',30)

pause

% label markers by best fitting model

scatter([RL1_NonGreedy(RL1_dat.sam_ind_RFX)';RL2_NonGreedy(RL2_dat.OA.sam_ind_RFX)], [RL1_MI(RL1_dat.sam_ind_RFX);RL2_MI(RL2_dat.OA.sam_ind_RFX)'], 3000, [RL1_dat.RFX_max_plac(RL1_dat.sam_ind_RFX)';RL2_dat.OA.RFX_max(RL2_dat.OA.sam_ind_RFX)'],'.');
colorbar
caxis([1 3])
colormap(jet)
title('OA Model attribution')
xlabel('Fraction non-greedy')
ylabel('Mutual information')
set(gca,'FontSize',30)

pause

% label markers by best fitting model excluding noisyRL

scatter([RL1_NonGreedy(RL1_dat.sam_ind_RFX)';RL2_NonGreedy(RL2_dat.OA.sam_ind_RFX)], [RL1_MI(RL1_dat.sam_ind_RFX);RL2_MI(RL2_dat.OA.sam_ind_RFX)'], 3000, [RL1_dat.RFX_max_plac2(RL1_dat.sam_ind_RFX)';RL2_dat.OA.RFX_max2(RL2_dat.OA.sam_ind_RFX)'],'.');
colorbar
caxis([1 3])
colormap(jet)
title('OA Model attribution')
xlabel('Fraction non-greedy')
ylabel('Mutual information')
set(gca,'FontSize',30)

pause

% check what proportion of subjects would be mislabeled as RL+rep instead
% of noisyRL
missC=sum(([RL1_dat.RFX_max_plac(RL1_dat.sam_ind_RFX)';RL2_dat.OA.RFX_max(RL2_dat.OA.sam_ind_RFX)']==3 & [RL1_dat.RFX_max_plac2(RL1_dat.sam_ind_RFX)';RL2_dat.OA.RFX_max2(RL2_dat.OA.sam_ind_RFX)'] == 2));
missC_prop=missC/sum([RL1_dat.RFX_max_plac(RL1_dat.sam_ind_RFX)';RL2_dat.OA.RFX_max(RL2_dat.OA.sam_ind_RFX)']==3);
% 43% of noisyRL subjects misclassified as RL+rep

missC=sum(([RL1_dat.RFX_max_ldopa(RL1_dat.sam_ind_RFX)']==3 & [RL1_dat.RFX_max_ldopa2(RL1_dat.sam_ind_RFX)'] == 2));
missC_prop=missC/sum([RL1_dat.RFX_max_ldopa(RL1_dat.sam_ind_RFX)']==3);

% label by repetition bias of RL+rep model

scatter([RL1_NonGreedy(RL1_dat.sam_ind_RFX)';RL2_NonGreedy(RL2_dat.OA.sam_ind_RFX)], [RL1_MI(RL1_dat.sam_ind_RFX);RL2_MI(RL2_dat.OA.sam_ind_RFX)'], 3000, [RL1_dat.par_RL_rep(RL1_dat.sam_ind_RFX,4,1);RL2_dat.OA.par_RL_rep(RL2_dat.OA.sam_ind_RFX,4)],'.');
colorbar
caxis([-max(RL1_dat.par_RL_rep(RL1_dat.sam_ind_RFX,4,1)') max(RL1_dat.par_RL_rep(RL1_dat.sam_ind_RFX,4,1)')])
colormap(hot_cool_c)
title('OA RL+rep rep bias')
xlabel('Fraction non-greedy')
ylabel('Mutual information')
set(gca,'FontSize',30)

pause

scatter([NonGreedy(RL2_dat.YA.sam_ind_RFX)], [MI_subs(RL2_dat.YA.sam_ind_RFX)'], 3000, [RL2_dat.YA.par_RL_rep(RL2_dat.YA.sam_ind_RFX,4)],'.');
colorbar
caxis([-max(RL2_dat.YA.par_RL_rep(RL2_dat.YA.sam_ind_RFX,4)') max(RL2_dat.YA.par_RL_rep(RL2_dat.YA.sam_ind_RFX,4)')])
colormap(hot_cool_c)
title('YA RL+rep rep bias')
xlabel('Fraction non-greedy')
ylabel('Mutual information')
set(gca,'FontSize',30)

% %% Evaluate learning noise contribution in non-greedy trials
% load('/Volumes/fb-lip/Projects/Naftali/data/analysis/PLS/figures/Z_archive/pre_commonCoord_correction/custom_colorbars/CustomColors.mat','hot_cool_c')
% 
% load('/Users/skowron/Documents/Suboptimality_models/learning_variability-master/tradeoff_pers_deBoer_YA.mat','NonGreedy_trials','highQ_RL')
% load('/Users/skowron/Documents/Suboptimality_models/learning_variability-master/highQ_noisyRL_deBoer_YA.mat')
% NonGreedy_trials_YA=NonGreedy_trials;
% highQ_RL_YA=highQ_RL;
% highQ_noisy_YA=highQ_noisy;
% clear NonGreedy_trials highQ_R highQ_noisy
% 
% % prepare output
% NoisyNonGreedy_fract_YA=zeros(length(ID_YA));
% 
% for sub=1:length(ID_YA)
% 
%     Qrev=highQ_RL_YA{sub}(NonGreedy_trials_YA{sub})~=highQ_noisy_YA{sub}(NonGreedy_trials_YA{sub});
%     Qrev=Qrev(2:end); % don't consider first trial since choice cannot be classified as greedy or non-greedy
%     Qrev_fract=sum(Qrev)/length(Qrev); % fraction of non-greedy trials better explained by learning noise
% 
%     NoisyNonGreedy_fract_YA(sub)=Qrev_fract;
% 
%     clear Qrev Qrev_fract
% end
% 
% load('/Users/skowron/Documents/Suboptimality_models/learning_variability-master/tradeoff_pers_deBoer_OA.mat','NonGreedy_trials','highQ_RL')
% load('/Users/skowron/Documents/Suboptimality_models/learning_variability-master/highQ_noisyRL_deBoer_OA.mat')
% NonGreedy_trials_OA=NonGreedy_trials;
% highQ_RL_OA=highQ_RL;
% highQ_noisy_OA=highQ_noisy;
% clear NonGreedy_trials highQ_R highQ_noisy
% 
% % prepare output
% NoisyNonGreedy_fract_OA=zeros(length(ID_OA));
% 
% for sub=1:length(ID_OA)
% 
%     Qrev=highQ_RL_OA{sub}(NonGreedy_trials_OA{sub})~=highQ_noisy_OA{sub}(NonGreedy_trials_OA{sub});
%     Qrev=Qrev(2:end); % don't consider first trial since choice cannot be classified as greedy or non-greedy
%     Qrev_fract=sum(Qrev)/length(Qrev); % fraction of non-greedy trials better explained by learning noise
% 
%     NoisyNonGreedy_fract_OA(sub)=Qrev_fract;
% 
%     clear Qrev Qrev_fract
% end
%     
% % plot results
% 
% figure
% hold on
% bar(1,mean(NoisyNonGreedy_fract_YA(YA_sam_ind_RFX)),'w')
% scatter(ones(1,length(NoisyNonGreedy_fract_YA(YA_sam_ind_RFX))),NoisyNonGreedy_fract_YA(YA_sam_ind_RFX),3000,Ediff_YA,'.') %YA
% bar(2,mean(NoisyNonGreedy_fract_OA(OA_sam_ind_RFX)),'w')
% scatter(ones(1,length(NoisyNonGreedy_fract_OA(OA_sam_ind_RFX)))*2,NoisyNonGreedy_fract_OA(OA_sam_ind_RFX),3000,Ediff_OA,'.') %OA
% 
% title('Fraction non-greedy explained by noisyRL (by model evidence)')
% 
% colorbar
% colormap(hot_cool_c)
% caxis([-max(Ediff_OA) max(Ediff_OA)])
% 
% xlim([0 3])
% ylabel('Proportion learning noise')
% xlabel('Age group')
% set(gca,'FontSize',30)
% hold off
