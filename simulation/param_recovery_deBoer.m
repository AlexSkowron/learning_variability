% check parameter and model recoverability
clear all

cd '/Users/skowron/Volumes/tardis/skowron/aging_learning_var/learning_variability/simulation'

load('/Users/skowron/Volumes/tardis/skowron/deBoer_RL/TAB_for_Alex2.mat','subjects','group')

ID_OA = cellfun(@(x) [x '_OA'],subjects(group==2),'UniformOutput',false);
ID_YA = cellfun(@(x) [x '_YA'],subjects(group==1),'UniformOutput',false);

%remove excluded OA subjects
ID_OA(find(strcmp(ID_OA,'dad11_OA')))=[];
ID_OA(find(strcmp(ID_OA,'dad31_OA')))=[];
ID_OA(find(strcmp(ID_OA,'dad26_OA')))=[];

model_ID = {'param00001','param00101','param11010','param11011','param11111'}; % must map to model name
model_name = {'noiselessRL' 'noiselessRL_rep' 'noisyRL_argmax' 'noisyRL_softmax' 'noisyRL_softmax_rep'};
Npar = [3,4,4,4,5];

SimPath='/Users/skowron/Volumes/tardis/skowron/aging_learning_var/learning_variability/model_fit/results/simulation_deBoer/';

%% parameter recovery

for m = 1:length(model_name)
        
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
        true_param=zeros(length(ID),Npar(m));
        rec_param=zeros(length(ID),Npar(m));

        for sub = 1:length(ID)

                %simulated par
                load([SimPath ID{sub} '/' model_name{m} '/subj' ID{sub} '_simul1_2q_complete0_' model_ID{m} '_leaky1.mat'],'param')

                true_param(sub,:)=param;
                clear param

                %recovered par
                load([SimPath ID{sub} '/' model_name{m} '/2q_complete0_subj' ID{sub} '_rec_resInf1_map1_traj0_simul0_' model_ID{m} '_leaky1.mat'],'map')

                rec_param(sub,:)=map;
                clear map

        end
         
        %save(['deBoer_' model_name{m} '_' gr_label '_sim.mat'],'true_param','rec_param')

        % plot data
        for p = 1:Npar(m)

            if strcmp(model_name{m},'noisyRL_argmax') && p==3 % skipped fixed beta parameter for noisyRL argmax
                continue
            end

            load([SimPath ID{sub} '/' model_name{m} '/2q_complete0_subj' ID{sub} '_rec_resInf1_map1_traj0_simul0_' model_ID{m} '_leaky1.mat'],'param_names')

            scatter(true_param(:,p),rec_param(:,p))
            title([model_name{m} ' ' gr_label ' ' param_names(p,:)])
            xlabel('true param')
            ylabel('recovered param')
            refline(1,0)
            
            if p == 3
                xlim([0,30])
                ylim([0,30])
            end
            
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

        clear true_param rec_param
    end
end