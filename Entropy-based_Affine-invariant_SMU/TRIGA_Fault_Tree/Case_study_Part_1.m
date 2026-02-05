%% Case Study Part 1: Generating the ECDF of the failure probability
%
% The codes provided here describe the approach to generate the N_obs observation of the failure probabilities p_f for the respective component

clc; clear;
%% Define the distribution of p_f for each Root event:

N_obs = 20; % No. of observations

R1 = betarnd(1.50, 43.00, N_obs, 1);
R2 = 3.45e-05;
R3 = betarnd(1.50, 43.00, N_obs, 1);
R4 = 3.45e-05;
R5 = betarnd(1.50, 62.40, N_obs, 1);
R6 = betarnd(1.50, 62.40, N_obs, 1);

%% Plot the ECDFs:

f = 20;
lab = {'Root event 1', 'Root event 2', 'Root event 3', 'Root event 4', 'Root event 5', 'Root event 6'};
data_cell = {R1, R2, R3, R4, R5, R6};

figure; 
hold on; box on; grid on;
for i = 1:6
subplot(2,3,i)
hold on; box on; grid on;
samp = data_cell{i};
[y1,x1] = ecdf(samp); stairs(x1, y1, 'r', 'LineWidth', 2);
ylim([0,1]); ylabel('Probability', 'FontName', 'Times'); xlabel('$p_{f}$', 'Interpreter', 'latex'); 
title(lab{i}, 'FontName', 'Times'); set(gca, 'Fontsize', f); xlim([0, 1.05*max(samp)])
end

%% Save data:
save('Case_Study_Part_1')
