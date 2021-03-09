% check parameter and model recoverability
clear all

cd '/Users/skowron/Volumes/tardis/skowron/aging_learning_var/learning_variability/simulation'

Nsub=1000;

model_ID = {'param11011','param11111'}; % must map to model name
model_name = {'noisyRL_softmax' 'noisyRL_softmax_rep'};
Npar = [4,5];

SimPath='/Users/skowron/Volumes/tardis/skowron/aging_learning_var/learning_variability/model_fit/results/sim_explore_all/';

%% parameter recovery

for m = 1:length(model_name)

        % retrieve data
        true_param=zeros(Nsub,Npar(m));
        rec_param=zeros(Nsub,Npar(m));

        for sub = 1:Nsub

                %simulated par
                load([SimPath '/' model_name{m} '/subj' num2str(sub) '_0_simul1_2q_complete0_' model_ID{m} '_leaky1.mat'],'param')
                
                true_param(sub,:)=param;
                clear param

                %recovered par
                load([SimPath '/' model_name{m} '/2q_complete0_subj' num2str(sub) '_0_resInf1_map1_traj0_simul0_' model_ID{m} '_leaky1.mat'],'map')

                rec_param(sub,:)=map;
                clear map

        end
        
        % subset of beta range
        if ~strcmp(model_name{m},'noisyRL_argmax')
            idx=true_param(:,3)>=5 & true_param(:,3)<=25;
            true_param=true_param(idx,:);
            rec_param=rec_param(idx,:);
        end
        
        % subset of eta range
        if contains(model_name{m},'rep')
            idx=true_param(:,end)>-5 & true_param(:,end)<5;
            true_param=true_param(idx,:);
            rec_param=rec_param(idx,:);
        end
     
        % subset alpha and epsilon range
        if contains(model_name{m},'noisyRL')
            idx=true_param(:,4)<0.4;
            true_param=true_param(idx,:);
            rec_param=rec_param(idx,:);
        end

        % plot data
        for p = 1:Npar(m)
               
%             if p==2 % skip alpha unchosen for fixes condition
%                 continue
%             end
            
            if strcmp(model_name{m},'noisyRL_argmax') && p==3 % skipped fixed beta parameter for noisyRL argmax
                continue
            end

            load([SimPath '/' model_name{m} '/2q_complete0_subj' num2str(sub) '_0_resInf1_map1_traj0_simul0_' model_ID{m} '_leaky1.mat'],'param_names')

            scatter(true_param(:,p),rec_param(:,p))
            title([model_name{m} ' ' param_names(p,:)])
            xlabel('true param')
            ylabel('recovered param')
            refline(1,0)
            
            if p==3
                ylim([0,100])
            end
            
            % Var explained by simulated pars
            [~,~,~,~,EXP]=regress(rec_param(:,p),[ones(size(true_param,1))', true_param]);
            EXP(1)
            
            pause       
            
%             % plot w/o outliers
%             sdOut=3; % std for outlier identification
% 
%             out_idx=true_param(:,p)>(median(true_param(:,p))+sdOut*std(true_param(:,p)))|true_param(:,p)<(median(true_param(:,p))-sdOut*std(true_param(:,p)));
% 
%             scatter(true_param(~out_idx,p),rec_param(~out_idx,p))
%             title(['<' num2str(sdOut) 'SD, N=' num2str(sum(~out_idx)) ' ' model_name{m} ' ' gr_label ' ' param_names(p,:)])
%             xlabel('true param')
%             ylabel('recovered param')
%             refline(1,0)
% 
%             pause

        
        end
        
        % plot correlations between recovered pars
        RecCorr=corr(rec_param,'type','Spearman')
        
        imagesc(RecCorr)
        colorbar
        title([model_name{m} ' recovered par correlations'])
        xticks(1:Npar(m))
        yticks(1:Npar(m))
        
        pause
        
        clear true_param rec_param RecCorr
end