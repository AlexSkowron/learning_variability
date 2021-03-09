%param recovery sim explore
clear all

cd /Users/skowron/Volumes/tardis/skowron/aging_learning_var/learning_variability/simulation

model_ID = {'param00001','param11010','param11011'}; % must map to model name
model_name = {'noiselessRL' 'noisyRL_argmax' 'noisyRL_softmax'};
Npar = [3,4,4];
Nsim = 30; % nr simulated subjects
Nite = 11; % nr of par settings

for m = 1:length(model_name)

    SimPath=['/Users/skowron/Volumes/tardis/skowron/aging_learning_var/learning_variability/model_fit/results/sim_explore/' model_name{m} '/'];
    
    for p=1:Npar(m)
        
        if strcmp(model_name(m),'noisyRL_argmax') && p == 3 % skip beta for argmax model
            continue
        end
        
        pam=p;
        
        if strcmp(model_name(m),'noisyRL_argmax') && p == 4 % rename par file name for argmax
            pam=3
        end
        
        % initialise
        rec_param=zeros(Nsim,Nite,Npar(m));
        true_param=zeros(1,Nite);
        
        for i=1:Nsim
           for ite = 1:Nite
               
               if strcmp(model_name(m),'noisyRL_argmax') && ite == 1 % skip epsilon=0 for argmax since model hard to fit (does not allow deviation from Q values)
                    continue
               end
               
               if i == 1
                   load([SimPath 'subj' num2str(i) '_' num2str(pam) '_' num2str(ite) '_simul1_2q_complete0_' model_ID{m} '_leaky1.mat'],'param')
                   true_param(ite)=param(p);
               end
               
%                % sim par
%                load(SimPath 'subj' num2str(i) '_' num2str(p) '_' num2str(ite) '_simul1_2q_complete0_param00001_leaky1.mat','param')
%                true_param(i,ite)=param(p);
%                clear param
               
               % rec par
               load([SimPath '2q_complete0_subj' num2str(i) '_' num2str(pam) '_' num2str(ite) '_resInf1_map1_traj0_simul0_' model_ID{m} '_leaky1.mat'],'map')
               rec_param(i,ite,:)=map;
               clear map
               
           end
        end
        
        % plot
        plot(rec_param(:,:,p)','bo')

        if max(max(rec_param(:,:,p))) > 200
            ylim([0,200])
            fprintf('y axis cutoff applied!')
        end
        
        hold on
        plot(true_param,'r.','MarkerSize',20)
        xlabel('param setting')
        ylabel('recovered param')
        title([model_name{m} ' par_manip' num2str(p)])
        
        hold off
        
        pause
        
        for pl = 1:Npar(m)
            
            if pl == p
                continue
            end
            
            if strcmp(model_name(m),'noisyRL_argmax') && pl == 3 % skip beta for argmax model
                continue
            end
            
            plot(rec_param(:,:,pl)','bo')

            if max(max(rec_param(:,:,pl))) > 200
                ylim([0,200])
                fprintf('y axis cutoff applied!')
            end

            xlabel('param setting')
            ylabel('recovered param')
            title([model_name{m} ' par_manip' num2str(p) ' par_view' num2str(pl)])

            pause

        end
        
        clear true_param rec_param
    end
end