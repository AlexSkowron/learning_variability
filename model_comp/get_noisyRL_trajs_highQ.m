% Evaluate proportion of non-greedy trials explained by learning noise
clear all

cd('/Users/skowron/Documents/Suboptimality_models/aging_learning_var/learning_variability/model_fit/results/fit_Chowdhury_RL_leaky1')

ID = {'01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32'};
drug_cond = {'placebo','ldopa'};
plot_traj=1; % toggle trajectory plotting for subjects

%% get Q value ranking from smoothing trajs for noisyRL
% prep output
highQ_noisy=cell(length(ID),length(drug_cond));

for drug=1:length(drug_cond)
    for sub=1:length(ID)
        
        load(['2q_complete0_subj' ID{sub} '_' drug_cond{drug} '_resInf1_map1_traj1_simul0_param11111_leaky1.mat'],'traj_noisy_guided')
        
        trajs=zeros(size(traj_noisy_guided,3),size(traj_noisy_guided,2));
        trajs(1,:)=mean(traj_noisy_guided(:,:,1));
        trajs(2,:)=mean(traj_noisy_guided(:,:,2));
        [~,highQ_noisy_sub]=max(trajs);
        
        highQ_noisy{sub,drug}=highQ_noisy_sub;
        
        % plot trajectories
            if plot_traj==1
            % Q1
            y = mean(traj_noisy_guided(:,:,1)); % your mean vector;
            x = 1:numel(y);
            std_dev = std(traj_noisy_guided(:,:,1));
            curve1 = y + std_dev;
            curve2 = y - std_dev;
            x2 = [x, fliplr(x)];
            inBetween = [curve1, fliplr(curve2)];
            fill(x2, inBetween, [0.17 0.17 0.17],'FaceAlpha',0.5);
            hold on;
            plot(x, y, 'k', 'LineWidth', 2);
            xlabel('trial')
            ylabel('Q-value')
            title(['sub-' ID{sub}])

            % Q2
            y = mean(traj_noisy_guided(:,:,2)); % your mean vector;
            x = 1:numel(y);
            std_dev = std(traj_noisy_guided(:,:,1));
            curve1 = y + std_dev;
            curve2 = y - std_dev;
            x2 = [x, fliplr(x)];
            inBetween = [curve1, fliplr(curve2)];
            fill(x2, inBetween, [0.17 0.17 0.17], 'FaceAlpha', 0.5);
            plot(x, y, 'k', 'LineWidth', 2);
            hold off

            pause
        end
        
        clear traj_noisy_guided trajs highQ_noisy_sub

    end
end

%save
save('/Users/skowron/Documents/Suboptimality_models/aging_learning_var/learning_variability/model_comp/results/highQ_noisyRL.mat','highQ_noisy')