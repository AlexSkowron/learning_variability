% check parameter and model recoverability
clear all

cd '/Users/skowron/Volumes/tardis/skowron/aging_learning_var/learning_variability/simulation'

ID = {'01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32'};

drug_cond = {'placebo','ldopa'};

model_ID = {'param00001','param00101','param11010','param11011','param11111'}; % must map to model name
model_name = {'noiselessRL' 'noiselessRL_rep' 'noisyRL_argmax' 'noisyRL_softmax' 'noisyRL_softmax_rep'};
Npar = [3,4,4,4,5];

SimPath='/Users/skowron/Volumes/tardis/skowron/aging_learning_var/learning_variability/model_fit/results/simulation_Chowdhury/';


%% parameter recovery
for m = 1:length(model_name)
    
    % retrieve data
    true_param=zeros(length(ID),Npar(m));
    rec_param=zeros(length(ID),Npar(m));
    
    for d = 1:length(drug_cond)
    
        for sub = 1:length(ID)

                % simulated par
                load([SimPath 'sub-' ID{sub} '_' drug_cond{d} '/' model_name{m} '/subj' ID{sub} '_' drug_cond{d} '_simul1_2q_complete0_' model_ID{m} '_leaky1.mat'],'param')

                true_param(sub,:)=param;
                clear param

                %recovered par
                load([SimPath 'sub-' ID{sub} '_' drug_cond{d} '/' model_name{m} '/2q_complete0_subj' ID{sub} '_' drug_cond{d} '_rec_resInf1_map1_traj0_simul0_' model_ID{m} '_leaky1.mat'],'map')

                rec_param(sub,:)=map;
                clear map

        end
         
        %save(['Chowdhury_' model_name{m} '_' drug_cond{d} '_sim.mat'],'true_param','rec_param')
    
        % plot data
        for p = 1:Npar(m)

            if strcmp(model_name{m},'noisyRL_argmax') && p==3 % skipped fixed beta parameter for noisyRL argmax
                continue
            end

            load([SimPath 'sub-' ID{sub} '_' drug_cond{d} '/' model_name{m} '/2q_complete0_subj' ID{sub} '_' drug_cond{d} '_rec_resInf1_map1_traj0_simul0_' model_ID{m} '_leaky1.mat'],'param_names')
            
            scatter(true_param(:,p),rec_param(:,p))
            title([model_name{m} ' ' drug_cond{d} ' ' param_names(p,:)])
            xlabel('true param')
            ylabel('recovered param')
            refline(1,0)
            
            if p == 3
                xlim([0,30])
                ylim([0,30])
            end
            
            pause
            
%             % plot w/o outliers
%             sdOut=2; % std for outlier identification
%             
%             out_idx=true_param(:,p)>(median(true_param(:,p))+sdOut*std(true_param(:,p)))|true_param(:,p)<(median(true_param(:,p))-sdOut*std(true_param(:,p)));
%             
%             scatter(true_param(~out_idx,p),rec_param(~out_idx,p))
%             title(['<' num2str(sdOut) 'SD, N=' num2str(sum(~out_idx)) ' ' model_name{m} ' ' drug_cond{d} ' ' param_names(p,:)])
%             xlabel('true param')
%             ylabel('recovered param')
%             refline(1,0)
%             
%             pause

        end
    
        clear true_param rec_param
    end
end