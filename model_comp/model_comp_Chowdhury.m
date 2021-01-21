% Perform Bayesian model comparison
clear all

addpath(genpath('/Users/skowron/Documents/MATLAB/VBA-toolbox-master'))

model_path='/Users/skowron/Documents/Suboptimality_models/aging_learning_var/learning_variability/model_fit/results/fit_Chowdhury_RL_leaky1/';

%Chowdhury data
ID = {'01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32'};

drug_cond = {'placebo','ldopa'};

model_ID = {'param00001','param00101','param11011','param11111'}; % standard RL, RL with perseveration, noisy RL, noisy RL with perseveration

% prepare output mat
Evidences=zeros(length(model_ID)+1,length(ID),length(drug_cond)); % models +1 for guessing model, see below

par_RL=zeros(length(ID),3,length(drug_cond));
par_RL_rep=zeros(length(ID),4,length(drug_cond));
par_RL_noisy=zeros(length(ID),4,length(drug_cond));
par_RL_noisy_rep=zeros(length(ID),5,length(drug_cond));

cd(model_path)

%% get data

for sub = 1:length(ID)
   for drug = 1:length(drug_cond)      
        for model = 1:length(model_ID)
        
            load(['2q_complete0_subj' ID{sub} '_' drug_cond{drug} '_resInf1_map1_traj1_simul0_' model_ID{model} '_leaky1.mat'],'results','map')
            
            Evidences(model,sub,drug)=results{end}(end);
            clear results
            
            if strcmp(model_ID{model},'param00001')
                par_RL(sub,:,drug)=map;
            elseif strcmp(model_ID{model},'param00101')
                par_RL_rep(sub,:,drug)=map;
            elseif strcmp(model_ID{model},'param11011')
                par_RL_noisy(sub,:,drug)=map;
            elseif strcmp(model_ID{model},'param11111')
                par_RL_noisy_rep(sub,:,drug)=map;
            else
                fprintf('error in map allocation\n')
            end
            
        end
       
   end
end

%% include model that assumes random guessing strategy
load('/Volumes/fb-lip/Projects/UCL_Chowdhury_RL/data/behaviour/beh_data/beh_data.mat','Ntrials_placebo','Ntrials_ldopa')

Evidences(end,:,1)=cell2mat(Ntrials_placebo)*log(0.5); % placebo cond
Evidences(end,:,2)=cell2mat(Ntrials_ldopa)*log(0.5); % ldopa cond

% save evidences
save('/Users/skowron/Documents/Suboptimality_models/aging_learning_var/learning_variability/model_comp/results/Evidences+pars','Evidences','par_RL','par_RL_rep','par_RL_noisy','par_RL_noisy_rep')

%% FFX analysis

% sum evidences across subjects and conditions
Esum_plac=sum(Evidences(:,:,1),2);
Esum_ldopa=sum(Evidences(:,:,2),2);
E_sum_all=sum(Evidences(:,:,:),2);
E_sum_all=sum(E_sum_all,3);

% plot
plot(Esum_plac)
title('Sum log evidence placebo')
xlabel('Model')

pause

plot(Esum_ldopa)
title('Sum log evidence Ldopa')
xlabel('Model')

pause

plot(E_sum_all)
title('Sum log evidence all')
xlabel('Model')

pause

% inspect model evidence variance

plot(Evidences(:,:,1),'bo','MarkerSize',5)
title('Log evidence placebo')
xlabel('Model')

pause

plot(Evidences(:,:,2),'bo','MarkerSize',5)
title('Log evidence Ldopa')
xlabel('Model')

pause

plot(sum(Evidences,3),'bo','MarkerSize',5)
title('Log evidence all')
xlabel('Model')

% compare model parameters on best fitting model (model 4 - noisy RL with pers)

for i = 1:length(model_ID)
    histogram(par_RL_noisy_rep(:,i,2)-par_RL_noisy_rep(:,i,1))
    [h,p]=ttest(par_RL_noisy_rep(:,i,2)-par_RL_noisy_rep(:,i,1))
    pause
end

%% RFX analysis

% get best model for each subject and condition
[~,maxE_plac] = max(Evidences(:,:,1));
[~,maxE_ldopa] = max(Evidences(:,:,2));

%plot

histogram(maxE_plac)
title('Best model frequency placebo')
xlabel('Model')

pause

histogram(maxE_ldopa)
title('Best model frequency Ldopa')
xlabel('Model')

pause

% compare model evidence for best model to chance model
% placebo
Ediff_best_chance_plac=zeros(1,length(ID));

for i = 1:length(ID)
    Ediff_best_chance_plac(i)=Evidences(maxE_plac(i),i,1);
end

Ediff_best_chance_plac=Ediff_best_chance_plac-(Evidences(5,:,1));

bar(Ediff_best_chance_plac)
title('Log evidence diff best-chance placebo')
xlabel('Subject')

pause

% ldopa
Ediff_best_chance_ldopa=zeros(1,length(ID));

for i = 1:length(ID)
    Ediff_best_chance_ldopa(i)=Evidences(maxE_ldopa(i),i,2);
end

Ediff_best_chance_ldopa=Ediff_best_chance_ldopa-(Evidences(5,:,2));

bar(Ediff_best_chance_ldopa)
title('Log evidence diff best-chance Ldopa')
xlabel('Subject')

pause

% --subject exclusion--
% Mark subjects where chance model fit best in either condition (excluded from analysis)
sam_ind=~(maxE_ldopa==5|maxE_plac==5); % 6

%subset
Evidences_sam=Evidences(1:4,sam_ind,:); % evidences for final sample!

save('/Users/skowron/Documents/Suboptimality_models/aging_learning_var/learning_variability/model_comp/results/Evidences+pars','Evidences_sam','sam_ind','-append')

% get model consistency
con=maxE_plac(sam_ind)==maxE_ldopa(sam_ind); % only 13 subjects show model consistency across conditions

%check model consistency separately for subgroups with best model fit unde
%rplac

con_best2=con(maxE_plac(sam_ind)==2); % 11/12 consistent
con_best3=con(maxE_plac(sam_ind)==3); % 1/9 consistent
con_best1=con(maxE_plac(sam_ind)==1); % 0/2
con_best4=con(maxE_plac(sam_ind)==4); % 0/3

% plot model transitions
plot([maxE_plac(sam_ind); maxE_ldopa(sam_ind)],'-bo')
title('best model transition')
ylabel('Model')
xlabel('drug cond')

pause

% plot model transitions for subgroups with best model fit under plac
plot([maxE_plac(sam_ind & maxE_plac==3); maxE_ldopa(sam_ind & maxE_plac==3)],'-bo')
title('best model transition for subgroup noisyRL')
ylabel('Model')
xlabel('drug cond')

pause

% compare model evidences for each condition
mod_leg={'RL' 'RL+rep' 'noisy RL' 'noisy RL+rep'};

%placebo

% sort evidences for plotting
sortE_plac=[Evidences_sam(:,maxE_plac(sam_ind)==3,1) Evidences_sam(:,maxE_plac(sam_ind)==2,1) Evidences_sam(:,maxE_plac(sam_ind)==4,1) Evidences_sam(:,maxE_plac(sam_ind)==1,1)];

%subtract subject mean for better plotting (will conceil between-subject differences)
sortE_plac=sortE_plac-mean(sortE_plac);

line_color = ['b' 'g' 'm' 'r'];

for i = 1:length(model_ID)
    plot(sortE_plac(i,:)',[line_color(i) '.'],'MarkerSize',30)
    hold on
end

%line([0:sum(sam_ind);0:sum(sam_ind)]+0.5,repmat(ylim',1,sum(sam_ind)+1),'Color','k')

legend(mod_leg)
title('Evidence comparison placebo')
ylabel('log model evidence (mean subtracted)')
xlabel('Subject')
hold off

pause

% model evidence comp placebo
% compare noisy RL and RL with rep bias fit directly
[Ediff_sorted_plac, Ei_plac]=sort(Evidences_sam(3,:,1)-Evidences_sam(2,:,1));
barh(Ediff_sorted_plac,'y','FaceAlpha',0.3) % noisy - rep
title('Placebo')
xlabel('Log model evidence difference')
ylabel('Subject')
set(gca,'FontSize',30)
xlim([-40,40])

pause

% plot model evidences for RL+rep to RL
E2diff_sorted_plac(Ei_plac)=Evidences_sam(1,:,1)-Evidences_sam(2,:,1);
barh(E2diff_sorted_plac,'c','FaceAlpha',0.3) % RL-rep
title('Placebo')
xlabel('Log model evidence difference')
ylabel('Subject')
set(gca,'FontSize',30)
xlim([-40,40])

% plot model evidences for noisyRL+rep to noisyRL
E3diff_sorted_plac(Ei_plac)=Evidences_sam(4,:,1)-Evidences_sam(3,:,1); %noisyRL+rep-noisyRL
barh(E3diff_sorted_plac,'m','FaceAlpha',0.3)
title('Placebo')
xlabel('Log model evidence difference')
ylabel('Subject')
set(gca,'FontSize',30)
xlim([-40,40])

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

pause

%ldopa

% sort evidences for plotting
sortE_ldopa=[Evidences_sam(:,maxE_ldopa(sam_ind)==3,2) Evidences_sam(:,maxE_ldopa(sam_ind)==2,2) Evidences_sam(:,maxE_ldopa(sam_ind)==4,2) Evidences_sam(:,maxE_ldopa(sam_ind)==1,2)];

%subtract subject mean for better plotting (will conceil between-subject differences)
sortE_ldopa=sortE_ldopa-mean(sortE_ldopa);

line_color = ['b' 'g' 'm' 'r'];

for i = 1:length(model_ID)
    plot(sortE_ldopa(i,:)',[line_color(i) '.'],'MarkerSize',30)
    hold on
end

legend(mod_leg)
title('Evidence comparison Ldopa')
ylabel('log model evidence (mean subtracted)')
xlabel('Subject')
hold off

pause

% model evidence comp ldopa
% compare noisy RL and RL with rep bias fit directly
[Ediff_sorted_ldopa, Ei_ldopa]=sort(Evidences_sam(3,:,2)-Evidences_sam(2,:,2));
barh(Ediff_sorted_ldopa,'y','FaceAlpha',0.3) % noisy - rep
title('L-Dopa')
xlabel('Log model evidence difference')
ylabel('Subject')
set(gca,'FontSize',30)
xlim([-40,40])

pause

% compare model evidences for RL+rep to RL
E2diff_sorted_ldopa(Ei_ldopa)=Evidences_sam(1,:,2)-Evidences_sam(2,:,2);
barh(E2diff_sorted_ldopa,'c','FaceAlpha',0.3) % RL-rep
title('L-Dopa')
xlabel('Log model evidence difference')
ylabel('Subject')
set(gca,'FontSize',30)
xlim([-40,40])

% compare model evidences for RL+rep to RL
E3diff_sorted_ldopa(Ei_ldopa)=Evidences_sam(4,:,2)-Evidences_sam(3,:,2); %noisyRL+rep-noisyRL
barh(E3diff_sorted_ldopa,'m','FaceAlpha',0.3)
title('L-Dopa')
xlabel('Log model evidence difference')
ylabel('Subject')
set(gca,'FontSize',30)
xlim([-40,40])

% % overlay model evidences for RL+rep to RL for noisyRL>RLrep subjects
% E2diff_sorted_ldopa(Ei_ldopa)=Evidences_sam(1,:,2)-Evidences_sam(2,:,2);
% E2diff_sorted_ldopa(Ediff_sorted_ldopa<0)=0;
% barh(E2diff_sorted_ldopa,'c','FaceAlpha',0.3) % RL-rep
% 
% pause
% 
% % overlay model evidences for noisyRL+rep to noisyRL for noisyRL>RLrep subjects
% E3diff_sorted_ldopa(Ei_ldopa)=Evidences_sam(4,:,2)-Evidences_sam(3,:,2); %noisyRL+rep-noisyRL
% E3diff_sorted_ldopa(Ediff_sorted_ldopa<0)=0;
% barh(E3diff_sorted_ldopa,'m','FaceAlpha',0.3)
% hold off
% 
% pause

% perform RFX Bayesian group model comparison excluding model 5

% [posterior_plac,out_plac] = VBA_groupBMC(Evidences(:,:,1));
% pause
% [posterior_ldopa,out_ldopa] = VBA_groupBMC(Evidences(:,:,2));
% pause
[posterior_cond_sam,out_cond_sam] = VBA_groupBMC_btwConds(Evidences_sam);


pause

% RFX Bayesian group model on all subjects and models
[posterior_cond,out_cond] = VBA_groupBMC_btwConds(Evidences);
[posterior_cond2,out_cond2] = VBA_groupBMC_btwConds(Evidences([1:2,5],:,:)); % model selection without noisy

% plot protected exceedance probability (likelihood that model is more frequent than any other model controlled for chance differences)
PEP_plac = (1-out_cond.VBA.cond(1).out.bor)*out_cond.VBA.cond(1).out.ep + out_cond.VBA.cond(1).out.bor/length(out_cond.VBA.cond(1).out.ep);
PEP_ldopa = (1-out_cond.VBA.cond(2).out.bor)*out_cond.VBA.cond(2).out.ep + out_cond.VBA.cond(2).out.bor/length(out_cond.VBA.cond(2).out.ep);
PEP_across = (1-out_cond.VBA.btw.out.bor)*out_cond.VBA.btw.out.ep + out_cond.VBA.btw.out.bor/length(out_cond.VBA.btw.out.ep);

mod_leg2=mod_leg;
mod_leg2{5}='Chance';

%placebo
bar(PEP_plac,0.95,'y')
ylim([0,1])
title('Placebo')
ylabel('Protected Exceedance Probability')
xlabel('Model')
xticklabels(mod_leg2)
hold on
plot(xlim,[0.95,0.95],'r-')
set(gca,'FontSize',18)
hold off

pause

%ldopa
bar(PEP_ldopa,0.95,'y')
ylim([0,1])
title('Ldopa')
ylabel('Protected Exceedance Probability')
xlabel('Model')
xticklabels(mod_leg2)
hold on
plot(xlim,[0.95,0.95],'r-')
set(gca,'FontSize',18)
hold off

pause

%across
bar(PEP_across,0.95,'y')
ylim([0,1])
title('Across drug condition')
ylabel('Protected Exceedance Probability')
xlabel('Model')
hold on
plot(xlim,[0.95,0.95],'r-')
set(gca,'FontSize',18)
hold off

pause

% plot model frequencies

%placebo
bar(out_cond.VBA.cond(1).out.Ef,0.95,'y')
ylim([0,1])
title('Placebo')
ylabel('Frequency')
xlabel('Model')
xticklabels(mod_leg2)
hold on
plot(xlim,repmat(1/length(out_cond.VBA.cond(1).out.Ef),1,2),'r-')
errorbar(out_cond.VBA.cond(1).out.Ef,sqrt(out_cond.VBA.cond(1).out.Vf),'k.')
set(gca,'FontSize',18)
hold off

pause

%ldopa
bar(out_cond.VBA.cond(2).out.Ef,0.95,'y')
ylim([0,1])
title('L-Dopa')
ylabel('Frequency')
xlabel('Model')
xticklabels(mod_leg2)
hold on
plot(xlim,repmat(1/length(out_cond.VBA.cond(2).out.Ef),1,2),'r-')
errorbar(out_cond.VBA.cond(2).out.Ef,sqrt(out_cond.VBA.cond(2).out.Vf),'k.')
set(gca,'FontSize',18)
hold off

pause

%across
bar(out_cond.VBA.btw.out.Ef,0.95,'y')
ylim([0,1])
title('Across drug condition')
ylabel('Frequency')
xlabel('Model')
hold on
plot(xlim,repmat(1/length(out_cond.VBA.btw.out.Ef),1,2),'r-')
errorbar(out_cond.VBA.btw.out.Ef,sqrt(out_cond.VBA.btw.out.Vf),'k.')
set(gca,'FontSize',18)
hold off

pause

%model attributions according to RFX model
[prob_plac,RFX_max_plac]=max(out_cond.VBA.cond(1).posterior.r); % prop gives the probability of the best model over all other models for a subject
[prob_ldopa,RFX_max_ldopa]=max(out_cond.VBA.cond(2).posterior.r);
[prob_change,RFX_max_change]=max(out_cond.VBA.btw.posterior.r); % transition tuple models

% would lead to exclusion of same subjects with best chance model
sam_ind_RFX=~(RFX_max_plac==5|RFX_max_ldopa==5);

%model attributions according to RFX model excluding noisyRL
[prob_plac2,RFX_max_plac2]=max(out_cond2.VBA.cond(1).posterior.r); % prop gives the probability of the best model over all other models for a subject
[prob_ldopa2,RFX_max_ldopa2]=max(out_cond2.VBA.cond(2).posterior.r);
[prob_change2,RFX_max_change2]=max(out_cond2.VBA.btw.posterior.r); % transition tuple models

save('/Users/skowron/Documents/Suboptimality_models/aging_learning_var/learning_variability/model_comp/results/Evidences+pars.mat','sam_ind_RFX','RFX_max_plac','RFX_max_ldopa','RFX_max_plac2','RFX_max_ldopa2','maxE_plac','maxE_ldopa','-append')

%plot posterior model probabilities
load('/Volumes/fb-lip/Projects/Naftali/data/analysis/PLS/figures/Z_archive/pre_commonCoord_correction/custom_colorbars/CustomColors.mat','hot_c')

%placebo
%sort for plotting
sorted_post_plac=out_cond.VBA.cond(1).posterior.r(:,sam_ind_RFX);
sorted_post_plac=[sorted_post_plac(:,RFX_max_plac(sam_ind_RFX)==1) sorted_post_plac(:,RFX_max_plac(sam_ind_RFX)==2) sorted_post_plac(:,RFX_max_plac(sam_ind_RFX)==3) sorted_post_plac(:,RFX_max_plac(sam_ind_RFX)==4) sorted_post_plac(:,RFX_max_plac(sam_ind_RFX)==5)];

imagesc(sorted_post_plac)
colorbar
colormap(hot_c)
xlabel('Subject')
ylabel('Model')
yticklabels(mod_leg2)
set(gca,'FontSize',30)
set(gca,'YTick',1:size(sorted_post_plac,1));
set(gca,'XTick',1:4:size(sorted_post_plac,2));
title('Posterior model probability Placebo')

pause


%plot results for model selection without noisyRL
sorted_post_plac2=out_cond2.VBA.cond(1).posterior.r(:,sam_ind_RFX);
sorted_post_plac2=[sorted_post_plac2(:,RFX_max_plac(sam_ind_RFX)==1) sorted_post_plac2(:,RFX_max_plac(sam_ind_RFX)==2) sorted_post_plac2(:,RFX_max_plac(sam_ind_RFX)==3) sorted_post_plac2(:,RFX_max_plac(sam_ind_RFX)==4) sorted_post_plac2(:,RFX_max_plac(sam_ind_RFX)==5)]; % sort according to other placebo plot to keep subject numbering consistent

imagesc(sorted_post_plac2)
colorbar
colormap(hot_c)
xlabel('Subject')
ylabel('Model')
yticklabels(mod_leg2([1:2 5]))
set(gca,'FontSize',30)
set(gca,'YTick',1:size(sorted_post_plac2,1));
set(gca,'XTick',1:4:size(sorted_post_plac2,2));
title('Posterior model probability Placebo')

pause

%ldopa
%sort for plotting
sorted_post_ldopa=out_cond.VBA.cond(2).posterior.r(:,sam_ind_RFX);
sorted_post_ldopa=[sorted_post_ldopa(:,RFX_max_ldopa(sam_ind_RFX)==1) sorted_post_ldopa(:,RFX_max_ldopa(sam_ind_RFX)==2) sorted_post_ldopa(:,RFX_max_ldopa(sam_ind_RFX)==3) sorted_post_ldopa(:,RFX_max_ldopa(sam_ind_RFX)==4) sorted_post_ldopa(:,RFX_max_ldopa(sam_ind_RFX)==5)];

imagesc(sorted_post_ldopa)
colorbar
colormap(hot_c)
xlabel('Subject')
ylabel('Model')
yticklabels(mod_leg2)
set(gca,'FontSize',30)
set(gca,'YTick',1:size(sorted_post_ldopa,1));
set(gca,'XTick',1:4:size(sorted_post_ldopa,2));
title('Posterior model probability L-Dopa')

pause

%plot results for model selection without noisyRL
sorted_post_ldopa2=out_cond2.VBA.cond(1).posterior.r(:,sam_ind_RFX);
sorted_post_ldopa2=[sorted_post_ldopa2(:,RFX_max_ldopa(sam_ind_RFX)==1) sorted_post_ldopa2(:,RFX_max_ldopa(sam_ind_RFX)==2) sorted_post_ldopa2(:,RFX_max_ldopa(sam_ind_RFX)==3) sorted_post_ldopa2(:,RFX_max_ldopa(sam_ind_RFX)==4) sorted_post_ldopa2(:,RFX_max_ldopa(sam_ind_RFX)==5)]; % sort according to other placebo plot to keep subject numbering consistent

imagesc(sorted_post_ldopa2)
colorbar
colormap(hot_c)
xlabel('Subject')
ylabel('Model')
yticklabels(mod_leg2([1:2 5]))
set(gca,'FontSize',30)
set(gca,'YTick',1:size(sorted_post_ldopa2,1));
set(gca,'XTick',1:4:size(sorted_post_ldopa2,2));
title('Posterior model probability L-Dopa')

pause

% plot posterior model probability for transition models - exclude all
% subjects that are best fitted by chance model at either occasion even if
% tuple models for those subjects best fit the chance model at either
% occasion!

%sort: first same family, then different family
sorted_post_change=[];
post_change_sam=out_cond.VBA.btw.posterior.r(:,sam_ind_RFX);

for s = [1,7,13,19,25,2:6,8:12,14:18,20:24]
    sorted_post_change=[sorted_post_change post_change_sam(:,RFX_max_change(sam_ind_RFX)==s)];
end

imagesc(sorted_post_change)
colorbar
colormap(hot_c)
xlabel('Subject')
ylabel('Model')
set(gca,'FontSize',30)
set(gca,'YTick',1:size(sorted_post_change,1));
set(gca,'XTick',1:4:size(sorted_post_change,2));
yticks([1:4:25])
title('Posterior model probability across drug conditions')

pause

%--compare parameters--

%RL_noisy with perseveration
load('2q_complete0_subj01_ldopa_resInf1_map1_traj1_simul0_param11111_leaky1.mat','param_names') % load param names

% plot parameter change (ldopa-placebo)
for l = 1:size(par_RL_noisy_rep,2)

    histogram(par_RL_noisy_rep(sam_ind,l,2)-par_RL_noisy_rep(sam_ind,l,1))
    title('noisyRL+rep')
    xlabel(param_names(l,:))

    pause
end

%test condition differences
for l = 1:size(par_RL_noisy_rep,2)

    [h,p]=ttest(par_RL_noisy_rep(sam_ind,l,2)-par_RL_noisy_rep(sam_ind,l,1))

    pause
end

% subgroup parameter comparison (effectively similar to a median split on epsilon/rep bias)

%specify if grouping according to individual log evidence or posterior prob
%from RFX analysis!!
Igroup=1

if Igroup == 1
    best_plac=RFX_max_plac;
    best_ldopa=RFX_max_ldopa;
elseif Igroup == 0
    best_plac=maxE_plac;
    best_ldopa=maxE_ldopa;
else
    fprintf('grouping variable not specified')
end

% group1: subjects best fit by model 3 or 4 under placebo
% plot parameter change (ldopa-placebo)
for l = 1:size(par_RL_noisy_rep,2)

    histogram(par_RL_noisy_rep(best_plac(sam_ind)==3 | best_plac(sam_ind)==4,l,2)-par_RL_noisy_rep(best_plac(sam_ind)==3 | best_plac(sam_ind)==4,l,1))
    title('noisyRL+rep')
    xlabel(param_names(l,:))

    pause
end

%test condition differences
for l = 1:size(par_RL_noisy_rep,2)

    [h,p]=ttest(par_RL_noisy_rep(best_plac(sam_ind)==3 | best_plac(sam_ind)==4,l,2)-par_RL_noisy_rep(best_plac(sam_ind)==3 | best_plac(sam_ind)==4,l,1))

    pause
end

% check noisyRL model for this group

% plot parameter change (ldopa-placebo)
for l = 1:size(par_RL_noisy,2)

    histogram(par_RL_noisy(best_plac(sam_ind)==3 | best_plac(sam_ind)==4,l,2)-par_RL_noisy(best_plac(sam_ind)==3 | best_plac(sam_ind)==4,l,1))
    title('noisyRL')
    xlabel(param_names(l,:))

    pause
end

%test condition differences
for l = 1:size(par_RL_noisy,2)

    [h,p]=ttest(par_RL_noisy(best_plac(sam_ind)==3 | best_plac(sam_ind)==4,l,2)-par_RL_noisy(best_plac(sam_ind)==3 | best_plac(sam_ind)==4,l,1))

    pause
end

% check noisyRL for best fit by noisyRL only
% check noisyRL model for this group

% plot parameter change (ldopa-placebo)
for l = 1:size(par_RL_noisy,2)

    histogram(par_RL_noisy(best_plac(sam_ind)==3,l,2)-par_RL_noisy(best_plac(sam_ind)==3,l,1))
    title('noisyRL')
    xlabel(param_names(l,:))

    pause
end

%test condition differences
for l = 1:size(par_RL_noisy,2)

    [h,p]=ttest(par_RL_noisy(best_plac(sam_ind)==3,l,2)-par_RL_noisy(best_plac(sam_ind)==3,l,1))

    pause
end

% group2: subjects best fit by model 2 under placebo

% plot parameter change (ldopa-placebo)
for l = 1:size(par_RL_noisy_rep,2)

    histogram(par_RL_noisy_rep(best_plac(sam_ind)==2,l,2)-par_RL_noisy_rep(best_plac(sam_ind)==2,l,1))
    title('noisyRL+rep')
    xlabel(param_names(l,:))

    pause
end

%test condition differences
for l = 1:size(par_RL_noisy_rep,2)

    [h,p]=ttest(par_RL_noisy_rep(best_plac(sam_ind)==2,l,2)-par_RL_noisy_rep(best_plac(sam_ind)==2,l,1))

    pause
end

%define subgroups based on RFX across condition model fits and compare
%parameters on the respective model

%group 1: subjects best fit by model 2 across conditions
param_names_rep=param_names([1:3 5],:);

% plot parameter change (ldopa-placebo)
for l = 1:size(par_RL_rep,2)

    histogram(par_RL_rep(RFX_max_change(sam_ind)==7,l,2)-par_RL_rep(RFX_max_change(sam_ind)==7,l,1))
    title('RL+rep')
    xlabel(param_names_rep(l,:))

    pause
end

%test condition differences
for l = 1:size(par_RL_rep,2)

    [h,p]=ttest(par_RL_rep(RFX_max_change(sam_ind)==7,l,2)-par_RL_rep(RFX_max_change(sam_ind)==7,l,1))

    pause
end

%group 2: subjects best fit by model 3 across conditions
param_names_noisy=param_names(1:4,:);

% plot parameter change (ldopa-placebo)
for l = 1:size(par_RL_noisy,2)

    histogram(par_RL_noisy(RFX_max_change(sam_ind)==13,l,2)-par_RL_noisy(RFX_max_change(sam_ind)==13,l,1))
    title('noisyRL')
    xlabel(param_names_noisy(l,:))

    pause
end

%test condition differences
for l = 1:size(par_RL_rep,2)

    [h,p]=ttest(par_RL_noisy(RFX_max_change(sam_ind)==13,l,2)-par_RL_noisy(RFX_max_change(sam_ind)==13,l,1))

    pause
end

%group 3: subjects best fit by model 4 across conditions
% plot parameter change (ldopa-placebo)
for l = 1:size(par_RL_noisy_rep,2)

    histogram(par_RL_noisy_rep(RFX_max_change(sam_ind)==19,l,2)-par_RL_noisy_rep(RFX_max_change(sam_ind)==19,l,1))
    title('noisyRL+rep')
    xlabel(param_names(l,:))

    pause
end

%test condition differences
for l = 1:size(par_RL_rep,2)

    [h,p]=ttest(par_RL_noisy_rep(RFX_max_change(sam_ind)==19,l,2)-par_RL_noisy_rep(RFX_max_change(sam_ind)==19,l,1))

    pause
end

%group 4: subjects best fit by model 3 or 4 across conditions and compared
%on model4 pars

% plot parameter change (ldopa-placebo)
for l = 1:size(par_RL_noisy_rep,2)

    histogram(par_RL_noisy_rep(RFX_max_change(sam_ind)==13 | RFX_max_change(sam_ind)==19,l,2)-par_RL_noisy_rep(RFX_max_change(sam_ind)==13 | RFX_max_change(sam_ind)==19,l,1))
    title('noisyRL+rep')
    xlabel(param_names(l,:))

    pause
end

%test condition differences
for l = 1:size(par_RL_rep,2)

    [h,p]=ttest(par_RL_noisy_rep(RFX_max_change(sam_ind)==13 | RFX_max_change(sam_ind)==19,l,2)-par_RL_noisy_rep(RFX_max_change(sam_ind)==13 | RFX_max_change(sam_ind)==19,l,1))

    pause
end

% compare parameters between subgroups note: 
% note1: new subgroup definition. Only comparing best model fit 2 versus 3
% at each condition separately
% note2: groups selected on rep bias and epsilon. So comparison only
% sensible for alpha and beta parameters, otherwise double-dipping

%log transform beta par values !!!
param_names(3,1:16)='log softmax beta';
par_RL_noisy_rep(:,3,:)=log(par_RL_noisy_rep(:,3,:));

for d = 1:length(drug_cond)
    for l = 1:size(par_RL_noisy_rep,2)
        
        if d==1
            maxE=best_plac;
        elseif d ==2
            maxE=best_ldopa;
        else
            fprintf('error\n')
        end
        
        histogram(par_RL_noisy_rep(maxE(sam_ind)==3,l,d),10)
        hold on
        histogram(par_RL_noisy_rep(maxE(sam_ind)==2,l,d),10)
        title(['Subgroups ' drug_cond{d}])
        xlabel(param_names(l,:))
        hold off
        
        [h,p]=ttest2(par_RL_noisy_rep(maxE(sam_ind)==3,l,d),par_RL_noisy_rep(maxE(sam_ind)==2,l,d))

        pause
    end
end

%inspect sign of rep bias in RL+rep group
rep_group_plac=par_RL_noisy_rep(maxE_plac(sam_ind)==2,end,1); % 3<0 -> so most positive rep bias
rep_group_ldopa=par_RL_noisy_rep(maxE_ldopa(sam_ind)==2,end,2); % 3<0

%inspect sign of rep bias in noisyRL+rep group
rep_group_plac=par_RL_noisy_rep(maxE_plac(sam_ind)==4,end,1); % all neg.
rep_group_ldopa=par_RL_noisy_rep(maxE_ldopa(sam_ind)==4,end,2); % all pos.
% -> noisyRL+rep group not comparable in terms of rep bias direction across
% conditions

% correlate parameters to task performance

% exclude outliers !!!
% sam_ind(15)=0; % remove beta outlier
% sam_ind(13)=0; % remove money won outlier
% sam_ind(5)=0; % remove money won outlier

% money won
load('/Volumes/fb-lip/Projects/UCL_Chowdhury_RL/data/behaviour/task_performance/reward.mat')

money_won=zeros(2,sum(sam_ind));
money_won(1,:)=out_tab.money_won_placebo(sam_ind);
money_won(2,:)=out_tab.money_won_ldopa(sam_ind);

%plot corrs within cond
for d = 1:length(drug_cond)
    for l = 1:size(par_RL_noisy_rep,2)

        plot(par_RL_noisy_rep(sam_ind,l,d),money_won(d,:),'bo')
        lsline
        
        [r,p]=corr(par_RL_noisy_rep(sam_ind,l,d),money_won(d,:)');
        
        title(['noisyRL+rep ' drug_cond{d} ', r=' num2str(r) ', p=' num2str(p)])
        xlabel(param_names(l,:))
        ylabel('money won')

        pause
    end
end

% plot corrs in change
for l = 1:size(par_RL_noisy_rep,2)

    plot(par_RL_noisy_rep(sam_ind,l,2)-par_RL_noisy_rep(sam_ind,l,1),money_won(2,:)-money_won(1,:),'bo')
    lsline

    [r,p]=corr(par_RL_noisy_rep(sam_ind,l,2)-par_RL_noisy_rep(sam_ind,l,1),money_won(2,:)'-money_won(1,:)');

    title(['noisyRL+rep change, r=' num2str(r) ', p=' num2str(p)])
    xlabel(param_names(l,:))
    ylabel('money won')

    pause
end

%switches
load('/Volumes/fb-lip/Projects/UCL_Chowdhury_RL/data/behaviour/task_performance/switches.mat','switches_plac_prop','switches_ldopa_prop')

switches=zeros(2,sum(sam_ind));
switches(1,:)=switches_plac_prop(sam_ind);
switches(2,:)=switches_ldopa_prop(sam_ind);

%plot corrs within cond
for d = 1:length(drug_cond)
    for l = 1:size(par_RL_noisy_rep,2)

        a=scatter(par_RL_noisy_rep(sam_ind,l,d),switches(d,:),500,[0.8500, 0.3250, 0.0980],'filled','MarkerEdgeColor',[1 1 1])
        a.MarkerFaceAlpha = .7;
        hline=refline;
        hline.LineWidth=2;
        hline.Color=[0 0 0];
        set(gca,'FontSize',30)
        yticks([0:0.1:0.5])

        [r,p]=corr(par_RL_noisy_rep(sam_ind,l,d),switches(d,:)')
        [rho,p]=corr(par_RL_noisy_rep(sam_ind,l,d),switches(d,:)','type','Spearman')

        %title(['noisyRL+rep ' drug_cond{d} ', r=' num2str(r) ', p=' num2str(p)])
        if d ==1
            title('Placebo')
        elseif d==2
            title('L-Dopa')
        end
        xlabel(param_names(l,:))
        ylabel('Proportion of Switches')

        pause
    end
end

% plot correlations for subjects best fitted by model 3 or 4
for d = 1:length(drug_cond)
    for l = 1:size(par_RL_noisy_rep,2)
        
        if d==1
            Gind=best_plac==3|best_plac==4;
            Gind(~sam_ind)=0;
        elseif d==2
            Gind=best_ldopa==3|best_plac==4;
            Gind(~sam_ind)=0;
        end

        plot(par_RL_noisy_rep(Gind,l,d),switches(d,Gind(sam_ind)),'ko','MarkerSize',20)
        set(gca,'FontSize',30)
        yticks([0:0.1:0.5])
        lsline

        [r,p]=corr(par_RL_noisy_rep(Gind,l,d),switches(d,Gind(sam_ind))')

        %title(['noisyRL+rep ' drug_cond{d} ', r=' num2str(r) ', p=' num2str(p)])
        if d ==1
            title('Placebo')
        elseif d==2
            title('L-Dopa')
        end
        xlabel(param_names(l,:))
        ylabel('Proportion of Switches')

        pause
    end
end

% plot correlations for subjects best fitted by model 2
for d = 1:length(drug_cond)
    for l = 1:size(par_RL_noisy_rep,2)
        
        if d==1
            Gind=best_plac==2;
            Gind(~sam_ind)=0;
        elseif d==2
            Gind=best_ldopa==2;
            Gind(~sam_ind)=0;
        end

        plot(par_RL_noisy_rep(Gind,l,d),switches(d,Gind(sam_ind)),'ko','MarkerSize',12)
        set(gca,'FontSize',30)
        yticks([0:0.1:0.5])
        lsline

        [r,p]=corr(par_RL_noisy_rep(Gind,l,d),switches(d,Gind(sam_ind))')

        %title(['noisyRL+rep ' drug_cond{d} ', r=' num2str(r) ', p=' num2str(p)])
        if d ==1
            title('Placebo')
        elseif d==2
            title('L-Dopa')
        end
        xlabel(param_names(l,:))
        ylabel('Proportion of Switches')

        pause
    end
end

% plot corrs in change
for l = 1:size(par_RL_noisy_rep,2)

    plot(par_RL_noisy_rep(sam_ind,l,2)-par_RL_noisy_rep(sam_ind,l,1),switches(2,:)-switches(1,:),'bo')
    lsline

    [r,p]=corr(par_RL_noisy_rep(sam_ind,l,2)-par_RL_noisy_rep(sam_ind,l,1),switches(2,:)'-switches(1,:)');

    title(['noisyRL+rep change, r=' num2str(r) ', p=' num2str(p)])
    xlabel(param_names(l,:))
    ylabel('switches prop')

    pause
end

% objectively adaptive switches
load('/Volumes/fb-lip/Projects/UCL_Chowdhury_RL/data/behaviour/task_performance/Obj_adapt_switches.mat','obj_adapt_switches_perc_placebo','obj_adapt_switches_perc_ldopa')

obj_switches=zeros(2,sum(sam_ind));
obj_switches(1,:)=obj_adapt_switches_perc_placebo(sam_ind);
obj_switches(2,:)=obj_adapt_switches_perc_ldopa(sam_ind);

%plot corrs within cond
for d = 1:length(drug_cond)
    for l = 1:size(par_RL_noisy_rep,2)

        plot(par_RL_noisy_rep(sam_ind,l,d),obj_switches(d,:),'bo')
        lsline

        [r,p]=corr(par_RL_noisy_rep(sam_ind,l,d),obj_switches(d,:)');

        title(['noisyRL+rep ' drug_cond{d} ', r=' num2str(r) ', p=' num2str(p)])
        xlabel(param_names(l,:))
        ylabel('obj adaptive switches prop')

        pause
    end
end

% plot corrs in change
for l = 1:size(par_RL_noisy_rep,2)

    plot(par_RL_noisy_rep(sam_ind,l,2)-par_RL_noisy_rep(sam_ind,l,1),obj_switches(2,:)-obj_switches(1,:),'bo')
    lsline

    [r,p]=corr(par_RL_noisy_rep(sam_ind,l,2)-par_RL_noisy_rep(sam_ind,l,1),obj_switches(2,:)'-obj_switches(1,:)');

    title(['noisyRL+rep change, r=' num2str(r) ', p=' num2str(p)])
    xlabel(param_names(l,:))
    ylabel('obj adaptive switches prop')

    pause
end

% correlate within-subject fited rep bias in RL+rep model to epsilon in
% noisyRL model
par_RLrep_sam=par_RL_rep(sam_ind);
par_RLnoisy_sam=par_RL_noisy(sam_ind);

for d = 1:length(drug_cond)
   
    plot(par_RLrep_sam(:,4,d),par_RLnoisy_sam(:,4,d),'o')
    lsline
    xlabel('rep_bias')
    ylabel('epsilon')
    title(drug_cond{d})
    [r,p]=corr(par_RLrep_sam(:,4,d),par_RLnoisy_sam(:,4,d))
    
    pause
    
    %subgroup best fitted by model 3 or 4
    if d == 1
        plot(par_RLrep_sam(RFX_max_plac(sam_ind)==3 | RFX_max_plac(sam_ind)==4,4,d),par_RLnoisy_sam(RFX_max_plac(sam_ind)==3 | RFX_max_plac(sam_ind)==4,4,d),'o')
        lsline
        xlabel('rep_bias')
        ylabel('epsilon')
        title(['Subgroup model 3 or 4 ' drug_cond{d}])
        
        [r,p]=corr(par_RLrep_sam(RFX_max_plac(sam_ind)==3 | RFX_max_plac(sam_ind)==4,4,d),par_RLnoisy_sam(RFX_max_plac(sam_ind)==3 | RFX_max_plac(sam_ind)==4,4,d))

        pause
    elseif d ==2
        plot(par_RLrep_sam(RFX_max_plac(sam_ind)==3 | RFX_max_plac(sam_ind)==4,4,d),par_RLnoisy_sam(RFX_max_plac(sam_ind)==3 | RFX_max_plac(sam_ind)==4,4,d),'o')
        lsline
        xlabel('rep_bias')
        ylabel('epsilon')
        title(['Subgroup model 3 or 4 ' drug_cond{d}])
        
        [r,p]=corr(par_RLrep_sam(RFX_max_plac(sam_ind)==3 | RFX_max_plac(sam_ind)==4,4,d),par_RLnoisy_sam(RFX_max_plac(sam_ind)==3 | RFX_max_plac(sam_ind)==4,4,d))
        
        pause
    end
end

%% plot trade-off between fraction non-greedy/switches and MI across successive trials by model log evidence difference between noisyRL and RL+rep
load('/Volumes/fb-lip/Projects/Naftali/data/analysis/PLS/figures/Z_archive/pre_commonCoord_correction/custom_colorbars/CustomColors.mat','hot_cool_c')

Ediff_plac=Evidences_sam(3,:,1)-Evidences_sam(2,:,1); % pos in favor of noisyRL
Ediff_ldopa=Evidences_sam(3,:,2)-Evidences_sam(2,:,2);

load('/Users/skowron/Documents/Suboptimality_models/learning_variability-master/tradeoff_pers.mat')

for d=1:length(drug_cond)
    
    if d==1
        MI=MI_subs_plac;
        Ediff=Ediff_plac;
    elseif d==2
        MI=MI_subs_ldopa;
        Ediff=Ediff_ldopa;
    else
        fprintf('Error')
    end
    
    scatter(NonGreedy(sam_ind,d), MI(sam_ind), 3000, Ediff,'.');
    colorbar
    caxis([min(Ediff) -min(Ediff)])
    colormap(hot_cool_c)
    title([drug_cond{d} ' Model log evidence diff RLnoisy - RL+rep'])
    xlabel('Fraction non-greedy')
    ylabel('Mutual information')
    set(gca,'FontSize',30)
    
    pause
    
end

% for noisyRL-RL
Ediff2_plac=Evidences_sam(3,:,1)-Evidences_sam(1,:,1); % pos in favor of noisyRL
Ediff2_ldopa=Evidences_sam(3,:,2)-Evidences_sam(1,:,2);

for d=1:length(drug_cond)
    
    if d==1
        MI=MI_subs_plac;
        Ediff=Ediff2_plac;
    elseif d==2
        MI=MI_subs_ldopa;
        Ediff=Ediff2_ldopa;
    else
        fprintf('Error')
    end
    
    scatter(NonGreedy(sam_ind,d), MI(sam_ind), 3000, Ediff,'.');
    colorbar
    caxis([-max(Ediff) max(Ediff)])
    colormap(hot_cool_c)
    title([drug_cond{d} ' Model log evidence diff RLnoisy - RL'])
    xlabel('Fraction non-greedy')
    ylabel('Mutual information')
    set(gca,'FontSize',30)
    
    pause
    
end

% label markers by best fitting model

for d=1:length(drug_cond)
    
    if d==1
        MI=MI_subs_plac;
        RFX_attr=RFX_max_plac(sam_ind);
    elseif d==2
        MI=MI_subs_ldopa;
        RFX_attr=RFX_max_ldopa(sam_ind);
    else
        fprintf('Error')
    end
    
    scatter(NonGreedy(sam_ind,d), MI(sam_ind), 3000, RFX_attr,'.');
    colorbar
    caxis([1 4])
    colormap(jet)
    title([drug_cond{d} ' Model attribution'])
    xlabel('Fraction non-greedy')
    ylabel('Mutual information')
    set(gca,'FontSize',30)
    
    pause
    
end

% label markers by best fitting model (from model set excluding noisyRL)
% note different labeling

for d=1:length(drug_cond)
    
    if d==1
        MI=MI_subs_plac;
        RFX_attr=RFX_max_plac2(sam_ind);
    elseif d==2
        MI=MI_subs_ldopa;
        RFX_attr=RFX_max_ldopa2(sam_ind);
    else
        fprintf('Error')
    end
    
    scatter(NonGreedy(sam_ind,d), MI(sam_ind), 3000, RFX_attr,'.');
    colorbar
    caxis([1 3])
    colormap(jet)
    title([drug_cond{d} ' Model attribution'])
    xlabel('Fraction non-greedy')
    ylabel('Mutual information')
    set(gca,'FontSize',30)
    
    pause
    
end

%switches
for d=1:length(drug_cond)
    
    if d==1
        MI=MI_subs_plac;
    elseif d==2
        MI=MI_subs_ldopa;
    else
        fprintf('Error')
    end
    
    scatter(switches(d,:), MI(sam_ind), 3000, Ediff,'.');
    colorbar
    caxis([min(Ediff) -min(Ediff)])
    colormap(hot_cool_c)
    title([drug_cond{d} ' Model log evidence diff RLnoisy - RL+rep'])
    xlabel('Proportion of switches')
    ylabel('Mutual information')
    set(gca,'FontSize',30)
    
    pause
    
end

%money
for d=1:length(drug_cond)
    
    if d==1
        MI=MI_subs_plac;
    elseif d==2
        MI=MI_subs_ldopa;
    else
        fprintf('Error')
    end
    
    scatter(money_won(d,:), MI(sam_ind), 3000, Ediff,'.');
    colorbar
    caxis([min(Ediff) -min(Ediff)])
    colormap(hot_cool_c)
    title([drug_cond{d} ' Model log evidence diff RLnoisy - RL+rep'])
    xlabel('Money won')
    ylabel('Mutual information')
    set(gca,'FontSize',30)
    
    pause
    
end

%% Evaluate learning noise contribution in non-greedy trials
load('/Volumes/fb-lip/Projects/Naftali/data/analysis/PLS/figures/Z_archive/pre_commonCoord_correction/custom_colorbars/CustomColors.mat','hot_cool_c')

load('/Users/skowron/Documents/Suboptimality_models/learning_variability-master/tradeoff_pers.mat','NonGreedy_trials','highQ_RL')
load('/Users/skowron/Documents/Suboptimality_models/learning_variability-master/highQ_noisyRL.mat')

% prepare output
NoisyNonGreedy_fract=zeros(length(ID),length(drug_cond));

for drug=1:length(drug_cond)
    for sub=1:length(ID)
           
        Qrev=highQ_RL{sub,drug}(NonGreedy_trials{sub,drug})~=highQ_noisy{sub,drug}(NonGreedy_trials{sub,drug});
        Qrev=Qrev(2:end); % don't consider first trial since choice cannot be classified as greedy or non-greedy
        Qrev_fract=sum(Qrev)/length(Qrev); % fraction of non-greedy trials better explained by learning noise
        
        NoisyNonGreedy_fract(sub,drug)=Qrev_fract;
        
        clear Qrev Qrev_fract
    end
end

% plot results
load('/Users/skowron/Documents/Suboptimality_models/learning_variability-master/fit_Chowdhury_RL_leaky1/Evidences+pars.mat','sam_ind') % final sample

figure
hold on
bar(1,mean(NoisyNonGreedy_fract(sam_ind,1)),'w')
scatter(ones(1,length(NoisyNonGreedy_fract(sam_ind,1))),NoisyNonGreedy_fract(sam_ind,1),3000,Ediff_plac,'.') %placebo
bar(2,mean(NoisyNonGreedy_fract(sam_ind,2)),'w')
scatter(ones(1,length(NoisyNonGreedy_fract(sam_ind,2)))*2,NoisyNonGreedy_fract(sam_ind,2),3000,Ediff_ldopa,'.') %ldopa

title('Fraction non-greedy explained by noisyRL (by model evidence)')

colorbar
colormap(hot_cool_c)
caxis([min(Ediff_ldopa) -min(Ediff_ldopa)])

xlim([0 3])
ylabel('Proportion learning noise')
xlabel('Drug condition')
set(gca,'FontSize',30)
hold off