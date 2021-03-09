% Perform Bayesian model comparison
clear all

addpath(genpath('/Users/skowron/Documents/MATLAB/VBA-toolbox-master'))

model_path='/Users/skowron/Volumes/tardis/skowron/aging_learning_var/learning_variability/model_fit/results/fit_deBoer_RL_weber0/';

%deBoer data
load('/Users/skowron/Volumes/tardis/skowron/deBoer_RL/TAB_for_Alex2.mat','subjects','group')

ID_OA = cellfun(@(x) [x '_OA'],subjects(group==2),'UniformOutput',false);
ID_YA = cellfun(@(x) [x '_YA'],subjects(group==1),'UniformOutput',false);

%remove excluded OA subjects
outOA=[find(strcmp(ID_OA,'dad11_OA')), find(strcmp(ID_OA,'dad31_OA')), find(strcmp(ID_OA,'dad26_OA'))];
ID_OA(outOA)=[];

model_ID = {'param01010','param01011','param01111'}; % noiselessRL, noiselessRL with perseveration, noisyRL argmax, noisyRL softmax, noisy RL with perseveration

ResultPath = '/Users/skowron/Volumes/tardis/skowron/aging_learning_var/learning_variability/model_comp/results/';

load('/Volumes/fb-lip/user/Alexander/Lieke_bandit_data/Ntrials.mat','Ntrials_YA','Ntrials_OA')
Ntrials_OA(outOA)=[];


%%

for age_gr = 1:2 % perform model comparison for each age group

if age_gr==1
    ID=ID_YA;
    gr_label='YA';
    Ntrials=Ntrials_YA;
elseif age_gr==2
    ID=ID_OA;
    gr_label='OA';
    Ntrials=Ntrials_OA;
else
    assert(0,'no age group specified for analysis')
end

% prepare output mat
Evidences=zeros(length(model_ID)+1,length(ID)); % models +1 for guessing model, see below

par_RL=zeros(length(ID),3);
par_RL_rep=zeros(length(ID),4);
par_RL_noisy_argmax=zeros(length(ID),3);
par_RL_noisy=zeros(length(ID),4);
par_RL_noisy_rep=zeros(length(ID),5);

cd(model_path)

%% get data
for sub = 1:length(ID)
    for model = 1:length(model_ID)
        
        load(['2q_complete0_subj' ID{sub} '_resInf1_map1_traj0_simul0_' model_ID{model} '_leaky1.mat'],'results','map')

        Evidences(model,sub)=results{end}(end);
        clear results
        
        if strcmp(model_ID{model},'param00001')
            par_RL(sub,:)=map;
        elseif strcmp(model_ID{model},'param00101')
            par_RL_rep(sub,:)=map;
        elseif strcmp(model_ID{model},'param01010')
            par_RL_noisy_argmax(sub,:)=map([1:2,4]);
        elseif strcmp(model_ID{model},'param01011')
            par_RL_noisy(sub,:)=map;
        elseif strcmp(model_ID{model},'param01111')
            par_RL_noisy_rep(sub,:)=map;
        else
            assert(0,'error in map allocation')
        end

    end
end

%% include model that assumes random guessing strategy
Evidences(end,:)=Ntrials*log(0.5);

% save evidences
save([ResultPath 'Evidences_param_deBoer_' gr_label '_weber0'],'Evidences','par_RL','par_RL_rep','par_RL_noisy','par_RL_noisy_rep')

%% FFX analysis

% sum evidences across subjects and conditions
Esum=sum(Evidences,2);

% plot
plot(Esum)
title(['Sum log evidence ' gr_label])
xlabel('Model')

pause

% inspect model evidence variance

plot(Evidences,'bo','MarkerSize',5)
title(['Log evidence placebo ' gr_label])
xlabel('Model')

pause

%% RFX analysis

% get best model for each subject and condition
[~,maxE] = max(Evidences);

%plot
histogram(maxE)
title(['Best model frequency ' gr_label])
xlabel('Model')

pause

% compare model evidence for best model to chance model
Ediff_best_chance=zeros(1,length(ID));

for i = 1:length(ID)
    Ediff_best_chance(i)=Evidences(maxE(i),i,1);
end

Ediff_best_chance=Ediff_best_chance-(Evidences(5,:));

bar(Ediff_best_chance)
title(['Log evidence diff best-chance ' gr_label])
xlabel('Subject')

pause

% --subject exclusion--
% Mark subjects where chance model fit best (excluded from analysis)
sam_ind=~(maxE==6); % 5

%subset
Evidences_sam=Evidences(1:5,sam_ind); % evidences for final sample!

%save
save([ResultPath 'Evidences_param_deBoer_' gr_label],'Evidences_sam','sam_ind','-append')

% compare model evidences
mod_leg={'RL' 'RL+rep' 'noisy RL argmax' 'noisy RL softmax' 'noisy RL+rep'};

% sort evidences for plotting
sortE=[Evidences_sam(:,maxE(sam_ind)==3) Evidences_sam(:,maxE(sam_ind)==2) Evidences_sam(:,maxE(sam_ind)==4) Evidences_sam(:,maxE(sam_ind)==1)];

%subtract subject mean for better plotting (will conceil between-subject differences)
sortE=sortE-mean(sortE);

line_color = ['b' 'g' 'm' 'r'];

for i = 1:length(model_ID)
    plot(sortE(i,:)',[line_color(i) '.'],'MarkerSize',30)
    hold on
end

%line([0:sum(sam_ind);0:sum(sam_ind)]+0.5,repmat(ylim',1,sum(sam_ind)+1),'Color','k')

legend(mod_leg)
title(['Evidence comparison ' gr_label])
ylabel('log model evidence (mean subtracted)')
xlabel('Subject')
hold off

pause

% model evidence comp
% compare noisy RL and RL with rep bias fit directly
[Ediff_sorted, Ei]=sort(Evidences_sam(3,:)-Evidences_sam(2,:));
barh(Ediff_sorted,'y','FaceAlpha',0.3) % noisy - rep
title(gr_label)
xlabel('Log model evidence difference')
ylabel('Subject')
set(gca,'FontSize',30)

if age_gr==1
    xlim([-20,20])
elseif age_gr==2
    xlim([-25,25])
end

pause

% plot model evidences for RL+rep to RL
E2diff_sorted(Ei)=Evidences_sam(1,:)-Evidences_sam(2,:);
barh(E2diff_sorted,'c','FaceAlpha',0.3) % RL-rep
title(gr_label)
xlabel('Log model evidence difference')
ylabel('Subject')
set(gca,'FontSize',30)

if age_gr==1
    xlim([-60,60])
elseif age_gr==2
    xlim([-160,160])
end

pause

% plot model evidences for noisyRL+rep to noisyRL
E3diff_sorted(Ei)=Evidences_sam(4,:)-Evidences_sam(3,:); %noisyRL+rep-noisyRL
barh(E3diff_sorted,'m','FaceAlpha',0.3)
title(gr_label)
xlabel('Log model evidence difference')
ylabel('Subject')
set(gca,'FontSize',30)

if age_gr==1
    xlim([-12,12])
elseif age_gr==2
    xlim([-15,15])
end

pause

% % overlay model evidences for RL+rep to RL for noisyRL>RLrep subjects
% E2diff_sorted_plac(Ei_plac)=Evidences_sam(1,:,1)-Evidences_sam(2,:,1);
% E2diff_sorted_plac(Ediff_sorted_plac<0)=0;
% barh(E2diff_sorted_plac,'c','FaceAlpha',0.3) % RL-rep
% 
% pause
% 
% % overlay model evidences for noisyRL+rep to noisyRL for noisyRL>RLrep subjects
% E3diff_sorted_plac(Ei_plac)=Evidences_sam(4,:,1)-Evidences_sam(3,:,1); %noisyRL+rep-noisyRL
% E3diff_sorted_plac(Ediff_sorted_plac<0)=0;
% barh(E3diff_sorted_plac,'m','FaceAlpha',0.3)
% hold off

% perform RFX Bayesian group model comparison excluding model 5

% [posterior_plac,out_plac] = VBA_groupBMC(Evidences(:,:,1));
% pause
% [posterior_ldopa,out_ldopa] = VBA_groupBMC(Evidences(:,:,2));
% pause

options.families = {[1:2], [3:5]}; % model families: noisy vs exact RL
[posterior_cond,out_cond] = VBA_groupBMC(Evidences(1:4,:),options);

[posterior_cond,out_cond] = VBA_groupBMC(Ea,options);

%[posterior_cond_sam,out_cond_sam] = VBA_groupBMC_btwGroups({EvidencesYA, EvidencesOA},options);

pause

% RFX Bayesian group model on all subjects and models

[posterior_cond,out_cond] = VBA_groupBMC(Evidences);
pause
[posterior_cond2,out_cond2] = VBA_groupBMC(Evidences([1:2,5],:)); % model selection without noisy
pause

% plot protected exceedance probability (likelihood that model is more frequent than any other model controlled for chance differences)
PEP = (1-out_cond.bor)*out_cond.ep + out_cond.bor/length(out_cond.ep);

mod_leg2=mod_leg;
mod_leg2{5}='Chance';

bar(PEP,0.95,'y')
ylim([0,1])
title(gr_label)
ylabel('Protected Exceedance Probability')
xlabel('Model')
xticklabels(mod_leg2)
hold on
plot(xlim,[0.95,0.95],'r-')
set(gca,'FontSize',18)
hold off

pause

% plot model frequencies

bar(out_cond.Ef,0.95,'y')
ylim([0,1])
title(gr_label)
ylabel('Frequency')
xlabel('Model')
xticklabels(mod_leg2)
hold on
plot(xlim,repmat(1/length(out_cond.Ef),1,2),'r-')
errorbar(out_cond.Ef,sqrt(out_cond.Vf),'k.')
set(gca,'FontSize',18)
hold off

pause

%model attributions according to RFX model
[prob,RFX_max]=max(posterior_cond.r); % prob gives the probability of the best model over all other models for a subject

% would lead to exclusion of same subjects with best chance model
sam_ind_RFX=~(RFX_max==5); % would exclude no subj for OA, same for YA as sam_ind

%model attributions according to RFX model excluding noisyRL
[prob2,RFX_max2]=max(posterior_cond2.r); % prop gives the probability of the best model over all other models for a subject

save(['/Users/skowron/Documents/Suboptimality_models/aging_learning_var/learning_variability/model_comp/results/Evidences+pars_' gr_label '.mat'],'sam_ind_RFX','RFX_max','RFX_max2','maxE','-append')

%plot posterior model probabilities
load('/Volumes/fb-lip/Projects/Naftali/data/analysis/PLS/figures/Z_archive/pre_commonCoord_correction/custom_colorbars/CustomColors.mat','hot_c')

%sort for plotting
sorted_post=posterior_cond.r(:,sam_ind_RFX);
sorted_post=[sorted_post(:,RFX_max(sam_ind_RFX)==1) sorted_post(:,RFX_max(sam_ind_RFX)==2) sorted_post(:,RFX_max(sam_ind_RFX)==3) sorted_post(:,RFX_max(sam_ind_RFX)==4) sorted_post(:,RFX_max(sam_ind_RFX)==5)];

imagesc(sorted_post)
colorbar
colormap(hot_c)
xlabel('Subject')
ylabel('Model')
yticklabels(mod_leg2)
set(gca,'FontSize',30)
set(gca,'YTick',1:size(sorted_post,1));
set(gca,'XTick',1:4:size(sorted_post,2));
title(['Posterior model probability ' gr_label])

pause

%plot results for model selection without noisyRL
sorted_post2=posterior_cond2.r(:,sam_ind_RFX);
sorted_post2=[sorted_post2(:,RFX_max(sam_ind_RFX)==1) sorted_post2(:,RFX_max(sam_ind_RFX)==2) sorted_post2(:,RFX_max(sam_ind_RFX)==3) sorted_post2(:,RFX_max(sam_ind_RFX)==4) sorted_post2(:,RFX_max(sam_ind_RFX)==5)]; % sort according to other placebo plot to keep subject numbering consistent

imagesc(sorted_post2)
colorbar
colormap(hot_c)
xlabel('Subject')
ylabel('Model')
yticklabels(mod_leg2([1:2 5]))
set(gca,'FontSize',30)
set(gca,'YTick',1:size(sorted_post2,1));
set(gca,'XTick',1:4:size(sorted_post2,2));
title(['Posterior model probability ' gr_label])

pause

end

%%
%--compare groups--
clear par_RL par_RL_noisy par_RL_noisy_rep sam_ind sam_ind_RFX

load('Evidences+pars_OA.mat','par_RL_rep','par_RL_noisy_rep','par_RL_noisy','sam_ind_RFX','RFX_max','RFX_max2','maxE','Evidences')
OA_par_RL_noisy_rep=par_RL_noisy_rep;
OA_par_RL_noisy=par_RL_noisy;
OA_par_RL_rep=par_RL_rep;
OA_sam_ind_RFX=sam_ind_RFX;
OA_Evidences=Evidences;
OA_RFX_max=RFX_max;
OA_RFX_max2=RFX_max2;
OA_maxE=maxE;
clear par_RL_noisy_rep sam_ind_RFX Evidences RFX_max maxE par_RL_noisy par_RL_rep RFX_max2

load([ResultPath 'Evidences_param_deBoer_YA.mat'],'par_RL_rep','par_RL_noisy_rep','par_RL_noisy','sam_ind_RFX','RFX_max2','RFX_max','maxE','Evidences')
YA_par_RL_noisy_rep=par_RL_noisy_rep;
YA_par_RL_noisy=par_RL_noisy;
YA_par_RL_rep=par_RL_rep;
YA_sam_ind_RFX=sam_ind_RFX;
YA_Evidences=Evidences;
YA_RFX_max=RFX_max;
YA_RFX_max2=RFX_max2;
YA_maxE=maxE;
clear par_RL_noisy_rep sam_ind_RFX Evidences RFX_max maxE par_RL_noisy par_RL_rep RFX_max2

% test if model frequencies differ between groups
[h,p]=VBA_groupBMC_btwGroups({YA_Evidences,OA_Evidences})

pause

%RL_noisy with perseveration
load('2q_complete0_subjdad02_OA_resInf1_map1_traj1_simul0_param11111_leaky1.mat','param_names') % load param names

% log transform beta!!
OA_par_RL_noisy_rep(:,3)=log(OA_par_RL_noisy_rep(:,3));
OA_par_RL_noisy(:,3)=log(OA_par_RL_noisy(:,3));
OA_par_RL_rep(:,3)=log(OA_par_RL_rep(:,3));

YA_par_RL_noisy_rep(:,3)=log(YA_par_RL_noisy_rep(:,3));
YA_par_RL_noisy(:,3)=log(YA_par_RL_noisy(:,3));
YA_par_RL_rep(:,3)=log(YA_par_RL_rep(:,3));

param_names(3,:)='log_beta      ';

% plot parameter difference between groups
for l = 1:size(OA_par_RL_noisy_rep,2)

    histogram(OA_par_RL_noisy_rep(OA_sam_ind_RFX,l),10) % blue OA
    hold on
    histogram(YA_par_RL_noisy_rep(YA_sam_ind_RFX,l),10) % orange YA
    hold off
    title('noisyRL+rep')
    xlabel(param_names(l,:))

    pause
end

%test group differences
for l = 1:size(OA_par_RL_noisy_rep,2)

    [h,p]=ttest2(OA_par_RL_noisy_rep(OA_sam_ind_RFX,l),YA_par_RL_noisy_rep(YA_sam_ind_RFX,l))

    pause
end

% subgroup parameter comparison

%specify if grouping according to individual log evidence or posterior prob
%from RFX analysis!!
Igroup=1

if Igroup == 1
    OA_best=OA_RFX_max;
    YA_best=YA_RFX_max;
elseif Igroup == 0
    OA_best=OA_maxE;
    YA_best=YA_maxE;
else
    assert(0,'grouping variable not specified')
end

% group1: subjects best fit by model 3 or 4
% plot group differences in parameters
for l = 1:size(OA_par_RL_noisy_rep,2)

    histogram(OA_par_RL_noisy_rep(OA_best(OA_sam_ind_RFX)==3 | OA_best(OA_sam_ind_RFX)==4,l),10)
    hold on
    histogram(YA_par_RL_noisy_rep(YA_best(YA_sam_ind_RFX)==3 | YA_best(YA_sam_ind_RFX)==4,l),10)
    hold off
    title('noisyRL+rep')
    xlabel(param_names(l,:))

    pause
end

%test condition differences
for l = 1:size(OA_par_RL_noisy_rep,2)

    [h,p]=ttest2(OA_par_RL_noisy_rep(OA_best(OA_sam_ind_RFX)==3 | OA_best(OA_sam_ind_RFX)==4,l),YA_par_RL_noisy_rep(YA_best(YA_sam_ind_RFX)==3 | YA_best(YA_sam_ind_RFX)==4,l))

    pause
end

% check noisyRL model for this group
for l = 1:size(OA_par_RL_noisy,2)

    histogram(OA_par_RL_noisy(OA_best(OA_sam_ind_RFX)==3 | OA_best(OA_sam_ind_RFX)==4,l),10)
    hold on
    histogram(YA_par_RL_noisy(YA_best(YA_sam_ind_RFX)==3 | YA_best(YA_sam_ind_RFX)==4,l),10)
    hold off
    title('noisyRL+rep')
    xlabel(param_names(l,:))

    pause
end

%test condition differences
for l = 1:size(OA_par_RL_noisy,2)

    [h,p]=ttest2(OA_par_RL_noisy(OA_best(OA_sam_ind_RFX)==3 | OA_best(OA_sam_ind_RFX)==4,l),YA_par_RL_noisy(YA_best(YA_sam_ind_RFX)==3 | YA_best(YA_sam_ind_RFX)==4,l))

    pause
end

% group2: subjects best fit by model 2
% very small N!
for l = 1:size(OA_par_RL_noisy_rep,2)

    histogram(OA_par_RL_noisy_rep(OA_best(OA_sam_ind_RFX)==2,l),10)
    hold on
    histogram(YA_par_RL_noisy_rep(YA_best(YA_sam_ind_RFX)==2,l),10)
    hold off
    title('noisyRL+rep')
    xlabel(param_names(l,:))

    pause
end

%test group differences
for l = 1:size(OA_par_RL_noisy_rep,2)

    [h,p]=ttest2(OA_par_RL_noisy_rep(OA_best(OA_sam_ind_RFX)==2,l),YA_par_RL_noisy_rep(YA_best(YA_sam_ind_RFX)==2,l))

    pause
end

%inspect sign of rep bias in RL+rep group
rep_group_YA=OA_par_RL_noisy_rep(OA_maxE(OA_sam_ind_RFX)==2,end); % mostly pos
rep_group_OA=YA_par_RL_noisy_rep(YA_maxE(YA_sam_ind_RFX)==2,end); % mostly pos

% conditions

% correlate parameters to task performance

load('/Volumes/fb-lip/user/Alexander/Lieke_bandit_data/performance.mat')

% total reward earned

%plot corrs within cond
%OA
for l = 1:size(OA_par_RL_noisy_rep,2)

    plot(OA_par_RL_noisy_rep(OA_sam_ind_RFX,l),total_reward_OA(OA_sam_ind_RFX),'bo')
    lsline

    [r,p]=corr(OA_par_RL_noisy_rep(OA_sam_ind_RFX,l),total_reward_OA(OA_sam_ind_RFX)');

    title(['noisyRL+rep OA, r=' num2str(r) ', p=' num2str(p)])
    xlabel(param_names(l,:))
    ylabel('total reward')

    pause
end

%YA
for l = 1:size(YA_par_RL_noisy_rep,2)

    plot(YA_par_RL_noisy_rep(YA_sam_ind_RFX,l),total_reward_YA(YA_sam_ind_RFX),'bo')
    lsline

    [r,p]=corr(YA_par_RL_noisy_rep(YA_sam_ind_RFX,l),total_reward_YA(YA_sam_ind_RFX)');

    title(['noisyRL+rep YA, r=' num2str(r) ', p=' num2str(p)])
    xlabel(param_names(l,:))
    ylabel('total reward')

    pause
end

% plot correlations for subjects best fitted by model 3 for noisyRL
% model pars
% OA
for l = 1:size(OA_par_RL_noisy,2)

    a=scatter(OA_par_RL_noisy(OA_sam_ind_RFX & OA_best==3,l),total_reward_OA(OA_sam_ind_RFX & OA_best==3),500,[0.8500, 0.3250, 0.0980],'filled','MarkerEdgeColor',[1 1 1])
    a.MarkerFaceAlpha = .7;
    hline=refline;
    hline.LineWidth=2;
    hline.Color=[0 0 0];
    set(gca,'FontSize',30)
    yticks([0:0.1:0.5])

    [r,p]=corr(OA_par_RL_noisy(OA_sam_ind_RFX & OA_best==3,l),total_reward_OA(OA_sam_ind_RFX & OA_best==3)')
    [rho,p2]=corr(OA_par_RL_noisy(OA_sam_ind_RFX & OA_best==3,l),total_reward_OA(OA_sam_ind_RFX & OA_best==3)','type','Spearman')

    %title(['noisyRL+rep ' drug_cond{d} ', r=' num2str(r) ', p=' num2str(p)])
    title(['noisyRL OA r=' num2str(r) ', p=' num2str(p)])
    
    xlabel(param_names(l,:))
    ylabel('total reward')

    pause
end

% YA
for l = 1:size(YA_par_RL_noisy,2)

    a=scatter(YA_par_RL_noisy(YA_sam_ind_RFX & YA_best==3,l),total_reward_YA(YA_sam_ind_RFX & YA_best==3),500,[0.8500, 0.3250, 0.0980],'filled','MarkerEdgeColor',[1 1 1])
    a.MarkerFaceAlpha = .7;
    hline=refline;
    hline.LineWidth=2;
    hline.Color=[0 0 0];
    set(gca,'FontSize',30)
    yticks([0:0.1:0.5])

    [r,p]=corr(YA_par_RL_noisy(YA_sam_ind_RFX & YA_best==3,l),total_reward_YA(YA_sam_ind_RFX & YA_best==3)')
    [rho,p2]=corr(YA_par_RL_noisy(YA_sam_ind_RFX & YA_best==3,l),total_reward_YA(YA_sam_ind_RFX & YA_best==3)','type','Spearman')

    %title(['noisyRL+rep ' drug_cond{d} ', r=' num2str(r) ', p=' num2str(p)])
    title(['noisyRL YA r=' num2str(r) ', p=' num2str(p)])
    
    xlabel(param_names(l,:))
    ylabel('total reward')

    pause
end

%switches

% OA
for l = 1:size(OA_par_RL_noisy_rep,2)

    a=scatter(OA_par_RL_noisy_rep(OA_sam_ind_RFX,l),switches_prop_OA(OA_sam_ind_RFX),500,[0.8500, 0.3250, 0.0980],'filled','MarkerEdgeColor',[1 1 1])
    a.MarkerFaceAlpha = .7;
    hline=refline;
    hline.LineWidth=2;
    hline.Color=[0 0 0];
    set(gca,'FontSize',30)
    yticks([0:0.1:0.5])

    [r,p]=corr(OA_par_RL_noisy_rep(OA_sam_ind_RFX,l),switches_prop_OA(OA_sam_ind_RFX)')
    [rho,p2]=corr(OA_par_RL_noisy_rep(OA_sam_ind_RFX,l),switches_prop_OA(OA_sam_ind_RFX)','type','Spearman')

    %title(['noisyRL+rep ' drug_cond{d} ', r=' num2str(r) ', p=' num2str(p)])
    title(['noisyRL+rep OA r=' num2str(r) ', p=' num2str(p)])
    
    xlabel(param_names(l,:))
    ylabel('Proportion of Switches')

    pause
end

% YA
for l = 1:size(YA_par_RL_noisy_rep,2)

    a=scatter(YA_par_RL_noisy_rep(YA_sam_ind_RFX,l),switches_prop_YA(YA_sam_ind_RFX),500,[0.8500, 0.3250, 0.0980],'filled','MarkerEdgeColor',[1 1 1])
    a.MarkerFaceAlpha = .7;
    hline=refline;
    hline.LineWidth=2;
    hline.Color=[0 0 0];
    set(gca,'FontSize',30)
    yticks([0:0.1:0.5])

    [r,p]=corr(YA_par_RL_noisy_rep(YA_sam_ind_RFX,l),switches_prop_YA(YA_sam_ind_RFX)')
    [rho,p2]=corr(YA_par_RL_noisy_rep(YA_sam_ind_RFX,l),switches_prop_YA(YA_sam_ind_RFX)','type','Spearman')

    %title(['noisyRL+rep ' drug_cond{d} ', r=' num2str(r) ', p=' num2str(p)])
    title(['noisyRL+rep YA r=' num2str(r) ', p=' num2str(p)])
    
    xlabel(param_names(l,:))
    ylabel('Proportion of Switches')

    pause
end

% plot correlations for subjects best fitted by model 3 for noisyRL
% model pars
% OA
for l = 1:size(OA_par_RL_noisy,2)

    a=scatter(OA_par_RL_noisy(OA_sam_ind_RFX & OA_best==3,l),switches_prop_OA(OA_sam_ind_RFX & OA_best==3),500,[0.8500, 0.3250, 0.0980],'filled','MarkerEdgeColor',[1 1 1])
    a.MarkerFaceAlpha = .7;
    hline=refline;
    hline.LineWidth=2;
    hline.Color=[0 0 0];
    set(gca,'FontSize',30)
    yticks([0:0.1:0.5])

    [r,p]=corr(OA_par_RL_noisy(OA_sam_ind_RFX & OA_best==3,l),switches_prop_OA(OA_sam_ind_RFX & OA_best==3)')
    [rho,p2]=corr(OA_par_RL_noisy(OA_sam_ind_RFX & OA_best==3,l),switches_prop_OA(OA_sam_ind_RFX & OA_best==3)','type','Spearman')

    %title(['noisyRL+rep ' drug_cond{d} ', r=' num2str(r) ', p=' num2str(p)])
    title(['noisyRL OA r=' num2str(r) ', p=' num2str(p)])
    
    xlabel(param_names(l,:))
    ylabel('Proportion of Switches')

    pause
end

% YA
for l = 1:size(YA_par_RL_noisy,2)

    a=scatter(YA_par_RL_noisy(YA_sam_ind_RFX & YA_best==3,l),switches_prop_YA(YA_sam_ind_RFX & YA_best==3),500,[0.8500, 0.3250, 0.0980],'filled','MarkerEdgeColor',[1 1 1])
    a.MarkerFaceAlpha = .7;
    hline=refline;
    hline.LineWidth=2;
    hline.Color=[0 0 0];
    set(gca,'FontSize',30)
    yticks([0:0.1:0.5])

    [r,p]=corr(YA_par_RL_noisy(YA_sam_ind_RFX & YA_best==3,l),switches_prop_YA(YA_sam_ind_RFX & YA_best==3)')
    [rho,p2]=corr(YA_par_RL_noisy(YA_sam_ind_RFX & YA_best==3,l),switches_prop_YA(YA_sam_ind_RFX & YA_best==3)','type','Spearman')

    %title(['noisyRL+rep ' drug_cond{d} ', r=' num2str(r) ', p=' num2str(p)])
    title(['noisyRL YA r=' num2str(r) ', p=' num2str(p)])
    
    xlabel(param_names(l,:))
    ylabel('Proportion of Switches')

    pause
end

%objectively adaptive switches
load('/Volumes/fb-lip/user/Alexander/Lieke_bandit_data/performance.mat','obj_adapt_switches_N_OA','obj_adapt_switches_N_YA','obj_adapt_switches_prop_OA','obj_adapt_switches_prop_YA')

%OA
%plot corrs within cond
for l = 1:size(OA_par_RL_noisy_rep,2)

    plot(OA_par_RL_noisy_rep(OA_sam_ind_RFX,l),obj_adapt_switches_prop_OA(OA_sam_ind_RFX),'bo')
    lsline

    [r,p]=corr(OA_par_RL_noisy_rep(OA_sam_ind_RFX,l),obj_adapt_switches_prop_OA(OA_sam_ind_RFX)');

    title(['noisyRL+rep OA, r=' num2str(r) ', p=' num2str(p)])
    xlabel(param_names(l,:))
    ylabel('obj adaptive switches prop')

    pause
end

%YA
%plot corrs within cond
for l = 1:size(YA_par_RL_noisy_rep,2)

    plot(YA_par_RL_noisy_rep(YA_sam_ind_RFX,l),obj_adapt_switches_prop_YA(YA_sam_ind_RFX),'bo')
    lsline

    [r,p]=corr(YA_par_RL_noisy_rep(YA_sam_ind_RFX,l),obj_adapt_switches_prop_YA(YA_sam_ind_RFX)');

    title(['noisyRL+rep YA, r=' num2str(r) ', p=' num2str(p)])
    xlabel(param_names(l,:))
    ylabel('obj adaptive switches prop')

    pause
end

% correlate within-subject fitted rep bias in RL+rep model to epsilon in
% noisyRL model

%OA
plot(OA_par_RL_rep(OA_sam_ind_RFX,4),OA_par_RL_noisy(OA_sam_ind_RFX,4),'o')
lsline
xlabel('rep_bias')
ylabel('epsilon')

[r,p]=corr(OA_par_RL_rep(:,4),OA_par_RL_noisy(:,4))
title(['OA r=' num2str(r) ', p=' num2str(p)])

pause

%YA
plot(YA_par_RL_rep(YA_sam_ind_RFX,4),YA_par_RL_noisy(YA_sam_ind_RFX,4),'o')
lsline
xlabel('rep_bias')
ylabel('epsilon')

[r,p]=corr(YA_par_RL_rep(:,4),YA_par_RL_noisy(:,4))
title(['YA r=' num2str(r) ', p=' num2str(p)])

pause

%% plot trade-off between fraction non-greedy/switches and MI across successive trials by model log evidence difference between noisyRL and RL+rep
load('/Volumes/fb-lip/Projects/Naftali/data/analysis/PLS/figures/Z_archive/pre_commonCoord_correction/custom_colorbars/CustomColors.mat','hot_cool_c')

Ediff_OA=OA_Evidences(3,OA_sam_ind_RFX)-OA_Evidences(2,OA_sam_ind_RFX); % pos in favor of noisyRL
Ediff_YA=YA_Evidences(3,YA_sam_ind_RFX)-YA_Evidences(2,YA_sam_ind_RFX);

for d=1:2 % cycle over age groups
    
    if d==1
        load('/Users/skowron/Documents/Suboptimality_models/learning_variability-master/tradeoff_pers_deBoer_YA.mat','MI_subs','NonGreedy')
        MI=MI_subs;
        Ediff=Ediff_YA;
        sam_ind_RFX=YA_sam_ind_RFX;
        age_gr='YA';
        clear MI_subs
    elseif d==2
        load('/Users/skowron/Documents/Suboptimality_models/learning_variability-master/tradeoff_pers_deBoer_OA.mat','MI_subs','NonGreedy')
        MI=MI_subs;
        Ediff=Ediff_OA;
        sam_ind_RFX=OA_sam_ind_RFX;
        age_gr='OA';
        clear MI_subs
    else
        assert(0,'Error')
    end
    
    scatter(NonGreedy(sam_ind_RFX), MI(sam_ind_RFX), 3000, Ediff,'.');
    colorbar
    caxis([min(Ediff) -min(Ediff)])
    colormap(hot_cool_c)
    title([age_gr ' Model log evidence diff RLnoisy - RL+rep'])
    xlabel('Fraction non-greedy')
    ylabel('Mutual information')
    set(gca,'FontSize',30)
    
    pause
    
    clear NonGreedy
end

% for noisyRL-RL

Ediff2_OA=OA_Evidences(3,OA_sam_ind_RFX)-OA_Evidences(1,OA_sam_ind_RFX); % pos in favor of noisyRL
Ediff2_YA=YA_Evidences(3,YA_sam_ind_RFX)-YA_Evidences(1,YA_sam_ind_RFX);

for d=1:2 % cycle over age groups
    
    if d==1
        load('/Users/skowron/Documents/Suboptimality_models/learning_variability-master/tradeoff_pers_deBoer_YA.mat','MI_subs','NonGreedy')
        MI=MI_subs;
        Ediff=Ediff2_YA;
        sam_ind_RFX=YA_sam_ind_RFX;
        age_gr='YA';
        clear MI_subs
    elseif d==2
        load('/Users/skowron/Documents/Suboptimality_models/learning_variability-master/tradeoff_pers_deBoer_OA.mat','MI_subs','NonGreedy')
        MI=MI_subs;
        Ediff=Ediff2_OA;
        sam_ind_RFX=OA_sam_ind_RFX;
        age_gr='OA';
        clear MI_subs
    else
        assert(0,'Error')
    end
    
    scatter(NonGreedy(sam_ind_RFX), MI(sam_ind_RFX), 3000, Ediff,'.');
    colorbar
    caxis([min(Ediff) -min(Ediff)])
    colormap(hot_cool_c)
    title([age_gr ' Model log evidence diff RLnoisy - RL'])
    xlabel('Fraction non-greedy')
    ylabel('Mutual information')
    set(gca,'FontSize',30)
    
    pause
    
    clear NonGreedy
end

% label markers by best fitting model
for d=1:2 % cycle over age groups
    
    if d==1
        load('/Users/skowron/Documents/Suboptimality_models/learning_variability-master/tradeoff_pers_deBoer_YA.mat','MI_subs','NonGreedy')
        MI=MI_subs;
        sam_ind_RFX=YA_sam_ind_RFX;
        age_gr='YA';
        RFX_attr=YA_RFX_max(YA_sam_ind_RFX);
        clear MI_subs
    elseif d==2
        load('/Users/skowron/Documents/Suboptimality_models/learning_variability-master/tradeoff_pers_deBoer_OA.mat','MI_subs','NonGreedy')
        MI=MI_subs;
        sam_ind_RFX=OA_sam_ind_RFX;
        age_gr='OA';
        RFX_attr=OA_RFX_max(OA_sam_ind_RFX);
        clear MI_subs
    else
        assert(0,'Error')
    end
    
    scatter(NonGreedy(sam_ind_RFX), MI(sam_ind_RFX), 3000, RFX_attr,'.');
    colorbar
    caxis([1 3])
    colormap(jet)
    title([age_gr ' Model attribution'])
    xlabel('Fraction non-greedy')
    ylabel('Mutual information')
    set(gca,'FontSize',30)
    
    pause
    
    clear NonGreedy
end

% label markers by best fitting model (from model set excluding noisyRL)
% note different labeling
for d=1:2 % cycle over age groups
    
    if d==1
        load('/Users/skowron/Documents/Suboptimality_models/learning_variability-master/tradeoff_pers_deBoer_YA.mat','MI_subs','NonGreedy')
        MI=MI_subs;
        sam_ind_RFX=YA_sam_ind_RFX;
        age_gr='YA';
        RFX_attr=YA_RFX_max2(YA_sam_ind_RFX);
        clear MI_subs
    elseif d==2
        load('/Users/skowron/Documents/Suboptimality_models/learning_variability-master/tradeoff_pers_deBoer_OA.mat','MI_subs','NonGreedy')
        MI=MI_subs;
        sam_ind_RFX=OA_sam_ind_RFX;
        age_gr='OA';
        RFX_attr=OA_RFX_max2(OA_sam_ind_RFX);
        clear MI_subs
    else
        assert(0,'Error')
    end
    
    scatter(NonGreedy(sam_ind_RFX), MI(sam_ind_RFX), 3000, RFX_attr,'.');
    colorbar
    caxis([1 3])
    colormap(jet)
    title([age_gr ' Model attribution'])
    xlabel('Fraction non-greedy')
    ylabel('Mutual information')
    set(gca,'FontSize',30)
    
    pause
    
    clear NonGreedy
end

%switches
for d=1:2 % cycle over age groups
    
    if d==1
        load('/Users/skowron/Documents/Suboptimality_models/learning_variability-master/tradeoff_pers_deBoer_YA.mat','MI_subs','NonGreedy')
        MI=MI_subs;
        Ediff=Ediff_YA;
        sam_ind_RFX=YA_sam_ind_RFX;
        age_gr='YA';
        switches=switches_prop_YA;
        clear MI_subs
    elseif d==2
        load('/Users/skowron/Documents/Suboptimality_models/learning_variability-master/tradeoff_pers_deBoer_OA.mat','MI_subs','NonGreedy')
        MI=MI_subs;
        Ediff=Ediff_OA;
        sam_ind_RFX=OA_sam_ind_RFX;
        age_gr='OA';
        switches=switches_prop_OA;
        clear MI_subs
    else
        assert(0,'Error')
    end
    
    scatter(switches(sam_ind_RFX), MI(sam_ind_RFX), 3000, Ediff,'.');
    colorbar
    caxis([min(Ediff) -min(Ediff)])
    colormap(hot_cool_c)
    title([age_gr ' Model log evidence diff RLnoisy - RL+rep'])
    xlabel('Proportion Switches')
    ylabel('Mutual information')
    set(gca,'FontSize',30)
    
    pause
    
    clear NonGreedy
end

%money
for d=1:2 % cycle over age groups
    
    if d==1
        load('/Users/skowron/Documents/Suboptimality_models/learning_variability-master/tradeoff_pers_deBoer_YA.mat','MI_subs','NonGreedy')
        MI=MI_subs;
        Ediff=Ediff_YA;
        sam_ind_RFX=YA_sam_ind_RFX;
        age_gr='YA';
        money=total_reward_YA;
        clear MI_subs
    elseif d==2
        load('/Users/skowron/Documents/Suboptimality_models/learning_variability-master/tradeoff_pers_deBoer_OA.mat','MI_subs','NonGreedy')
        MI=MI_subs;
        Ediff=Ediff_OA;
        sam_ind_RFX=OA_sam_ind_RFX;
        age_gr='OA';
        money=total_reward_OA;;
        clear MI_subs
    else
        assert(0,'Error')
    end
    
    scatter(money(sam_ind_RFX), MI(sam_ind_RFX), 3000, Ediff,'.');
    colorbar
    caxis([min(Ediff) -min(Ediff)])
    colormap(hot_cool_c)
    title([age_gr ' Model log evidence diff RLnoisy - RL+rep'])
    xlabel('Total reward')
    ylabel('Mutual information')
    set(gca,'FontSize',30)
    
    pause
    
    clear NonGreedy
end

%% Evaluate learning noise contribution in non-greedy trials
load('/Volumes/fb-lip/Projects/Naftali/data/analysis/PLS/figures/Z_archive/pre_commonCoord_correction/custom_colorbars/CustomColors.mat','hot_cool_c')

load('/Users/skowron/Documents/Suboptimality_models/learning_variability-master/tradeoff_pers_deBoer_YA.mat','NonGreedy_trials','highQ_RL')
load('/Users/skowron/Documents/Suboptimality_models/learning_variability-master/highQ_noisyRL_deBoer_YA.mat')
NonGreedy_trials_YA=NonGreedy_trials;
highQ_RL_YA=highQ_RL;
highQ_noisy_YA=highQ_noisy;
clear NonGreedy_trials highQ_R highQ_noisy

% prepare output
NoisyNonGreedy_fract_YA=zeros(length(ID_YA));

for sub=1:length(ID_YA)

    Qrev=highQ_RL_YA{sub}(NonGreedy_trials_YA{sub})~=highQ_noisy_YA{sub}(NonGreedy_trials_YA{sub});
    Qrev=Qrev(2:end); % don't consider first trial since choice cannot be classified as greedy or non-greedy
    Qrev_fract=sum(Qrev)/length(Qrev); % fraction of non-greedy trials better explained by learning noise

    NoisyNonGreedy_fract_YA(sub)=Qrev_fract;

    clear Qrev Qrev_fract
end

load('/Users/skowron/Documents/Suboptimality_models/learning_variability-master/tradeoff_pers_deBoer_OA.mat','NonGreedy_trials','highQ_RL')
load('/Users/skowron/Documents/Suboptimality_models/learning_variability-master/highQ_noisyRL_deBoer_OA.mat')
NonGreedy_trials_OA=NonGreedy_trials;
highQ_RL_OA=highQ_RL;
highQ_noisy_OA=highQ_noisy;
clear NonGreedy_trials highQ_R highQ_noisy

% prepare output
NoisyNonGreedy_fract_OA=zeros(length(ID_OA));

for sub=1:length(ID_OA)

    Qrev=highQ_RL_OA{sub}(NonGreedy_trials_OA{sub})~=highQ_noisy_OA{sub}(NonGreedy_trials_OA{sub});
    Qrev=Qrev(2:end); % don't consider first trial since choice cannot be classified as greedy or non-greedy
    Qrev_fract=sum(Qrev)/length(Qrev); % fraction of non-greedy trials better explained by learning noise

    NoisyNonGreedy_fract_OA(sub)=Qrev_fract;

    clear Qrev Qrev_fract
end
    
% plot results

figure
hold on
bar(1,mean(NoisyNonGreedy_fract_YA(YA_sam_ind_RFX)),'w')
scatter(ones(1,length(NoisyNonGreedy_fract_YA(YA_sam_ind_RFX))),NoisyNonGreedy_fract_YA(YA_sam_ind_RFX),3000,Ediff_YA,'.') %YA
bar(2,mean(NoisyNonGreedy_fract_OA(OA_sam_ind_RFX)),'w')
scatter(ones(1,length(NoisyNonGreedy_fract_OA(OA_sam_ind_RFX)))*2,NoisyNonGreedy_fract_OA(OA_sam_ind_RFX),3000,Ediff_OA,'.') %OA

title('Fraction non-greedy explained by noisyRL (by model evidence)')

colorbar
colormap(hot_cool_c)
caxis([-max(Ediff_OA) max(Ediff_OA)])

xlim([0 3])
ylabel('Proportion learning noise')
xlabel('Age group')
set(gca,'FontSize',30)
hold off
