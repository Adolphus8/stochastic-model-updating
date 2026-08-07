%% Case Study Part 1: Generating the ECDF of the failure probability
%
% The codes provided here describe the approach to generate the N_obs observation of the failure probabilities p_f for the respective component

clc; clear;
%% Define the k-out-of-n C-box:

Nsamps = 100000;
KN = @(k,n) [betarnd(k, n - k + 1, Nsamps, 1), betarnd(k + 1, n - k, Nsamps, 1)];

%% Define the distribution of p_f for each Basic event:

R1 = [0.000301, 0.007143, 0.001044, 0.000106, 0.001323, 0.000604, 0.000446, 0.016575, 0.002600, 0.001200]';  % Basic event 1: Valve V-3 failed to open
R2 = KN(345, 100000);                     % Basic event 2: Operator failed to open valve V-3
R3 = [0.000301, 0.007143, 0.001044, 0.000106, 0.001323, 0.000604, 0.000446, 0.016575, 0.002600, 0.001200]';  % Basic event 3: Valve V-4 failed to open
R4 = KN(345, 100000);                     % Basic event 4: Operator failed to open valve V-4
R5 = KN(1, 42);                           % Basic event 5: Inlet pipe of V-3 breaks
R6 = KN(1, 42);                           % Basic event 6: Inlet pipe of V-4 breaks
R7 = KN(0, 42);                           % Basic event 7: Thermal column breaks
R8 = KN(1, 42);                           % Basic event 8: Radial beam port 1 breaks
R9 = KN(1, 42);                           % Basic event 9: Radial beam port 2 breaks
R10 = KN(0, 42);                          % Basic event 10: Tangential beam port 1 breaks
R11 = KN(0, 42);                          % Basic event 11: Tangential beam port 2 breaks
R12 = KN(1, 42);                          % Basic event 12: Reactor pool breaks

%% Plot the ECDFs and C-boxes:

f = 18;
lab = {'Basic event 1', 'Basic event 2', 'Basic event 3', 'Basic event 4', 'Basic event 5', 'Basic event 6', ...
       'Basic event 7', 'Basic event 8', 'Basic event 9', 'Basic event 10', 'Basic event 11', 'Basic event 12'};
data_cell = {R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12};

figure;
for i = [1,3]
subplot(3,4,i)
hold on; box on; grid on;
samp = data_cell{i};
[y1,x1] = ecdf(samp); stairs(x1, y1, 'r', 'LineWidth', 2);
ylim([0,1]); ylabel('Probability', 'FontName', 'Times'); xlabel('$p_{f}$', 'Interpreter', 'latex'); 
title(lab{i}, 'FontName', 'Times'); set(gca, 'Fontsize', f); xlim([2e-04, 1.05*max(samp)])
end

for i = [2,4,5:12]
subplot(3,4,i)
hold on; box on; grid on;
samp = data_cell{i};
[y1,x1] = ecdf(samp(:,1)); stairs(x1, y1, 'r', 'LineWidth', 2); [y1,x1] = ecdf(samp(:,2)); stairs(x1, y1, 'r', 'LineWidth', 2);
plot([min(samp(:,1)), min(samp(:,2))], [0, 0], 'r', 'LineWidth', 2); plot([max(samp(:,1)), max(samp(:,2))], [1, 1], 'r', 'LineWidth', 2); 
ylim([0,1]); ylabel('Probability', 'FontName', 'Times'); xlabel('$p_{f}$ $[yr^{-1}]$', 'Interpreter', 'latex'); 
title(lab{i}, 'FontName', 'Times'); set(gca, 'Fontsize', f);
end

%% Stochastic model updating parameters:
Nsim = 10;      % The number of stochastic model output realisations
N = 1000;       % Define the sample size from posterior
dm = 6;         % To select the Area metric as the distance function for the ABC procedure
width_par = 0.05; % To select the likelihood function width parameter for the ABC procedure

%% Perform Bayesian model updating via ABC on the p_f for Basic events 1 and 3:
% The updating is done by scaling the data by a factor of 1e+04 so as to avoid potential numerical instability due to small values.
% Note: Skewness and Kurtosis are unaffected by scaling.

bounds1 = [0, 0.02].*1e+04; % Bounds on the SDF
data = R1;

% Update the hyper-parameters of the SDF:
priorpdf_R1 = @(x) isfeasible(bounds1, x); priorrnd_R1 = @(N) prior_staircaseRV(N, 1, bounds1); 
logL_R1 = @(theta) loglikelihood(theta, data.*1e+04, bounds1, dm, width_par, Nsim);  

tic;
TEMCMC1 = TEMCMCsampler('nsamples', N, 'loglikelihood', logL_R1, 'priorpdf', priorpdf_R1, 'priorrnd', priorrnd_R1);
timeTEMCMC1 = toc; fprintf('Time elapsed by TEMCMC sampler = %1f sec \n', timeTEMCMC1); 
sample_R1 = TEMCMC1.samples;

% Plot the necessary figures:

% Compile the posterior samples to plot the histograms and Fuzzy-sets:
samples = sample_R1; samples(:,1) = samples(:,1).*1e-04; samples(:,2) = samples(:,2).*(1e-04).^2; % Re-scale the mean and variance back to original scale

% Plot the posterior histograms:
figure; nbin = 10; f = 20;
alpha = 40; ci_mat = zeros(4,2); % Credible interval matrix
label = {'$\mu$', '$m_{2}$', '$m_{3}/(m_{2})^{3/2}$', '$m_{4}/(m_{2})^{2}$'};
for j = 1:4
subplot(2,2,j)
hold on; box on; grid on;
histogram(samples(:,j), nbin); xlabel(label{j}, 'Interpreter', 'latex'); ylabel('Count', 'FontName', 'Times');
set(gca, 'Fontsize', f); ci_mat(j,:) = [prctile(samples(:,j), alpha), prctile(samples(:,j), 100-alpha)];
end

% Compute the P-box for the basic events:
Ne = 1000;                       % No. of epistemic realizations to generate from the epistemic space
Na = 10000;                      % No. of aleatory realizations from the SDF

ci_mat(1,:) = ci_mat(1,:).*1e+04; ci_mat(2,:) = ci_mat(2,:).*(1e+04).^2; % Re-scale the credible interval of mean and variance to the 1e+04 scale

SDF_model = @(alpha_cut) SDF_rnd_func(bounds1, alpha_cut, 2, Na, Ne);   % The SDF RNG model
out_put = DLMC(ci_mat(:,:), SDF_model); pbox = (out_put.pbox).*(1e-04); % Re-scale the p-box samples back to original scale

% Plot the P-box on the P_f of the basic events:
figure; f = 20; ylab = {'Probability'}; lab = {'Basic event 1 / Basic event 3'};
hold on; box on; grid on;
[f1,x1] = ecdf(data); stairs(x1, f1, 'r', 'linewidth', 2); [f1,x1] = ecdf(pbox(:,1)); [f2,x2] = ecdf(pbox(:,2));
stairs(x1, f1, 'b', 'linewidth', 2); stairs(x2, f2, 'b', 'linewidth', 2, 'handlevisibility', 'off'); 
plot([min(x1),min(x2)], [0,0], 'b', 'linewidth', 2, 'handlevisibility', 'off'); plot([max(x1),max(x2)], [1,1], 'b', 'linewidth', 2, 'handlevisibility', 'off');
set(gca, 'Fontsize', f); xlabel('$p_{f}$', 'Interpreter', 'latex'); ylabel(ylab, 'FontName', 'Times'); title(lab, 'FontName', 'Times'); 
legend('Data', 'Probability-box', 'linewidth', 2, 'location', 'southeast'); xlim([0, max(x2)].*1.10);

%% Save P-box data:
writematrix([0,0; pbox], 'pbox_R1_R3.csv'); % Export Pbox for Basic event as a .csv file with first row being [0,0] as the R programming file reads the first row as headers

%% Save data:
save('Case_Study_Part_1')
