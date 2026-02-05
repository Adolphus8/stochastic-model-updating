%% Case Study Part 1: Stochastic model updating step
%
% The codes provided here describe the approach to gperform stochastic model updating on the SDF distribution over the 
% failure probability p_f of the root events: 1, 3, 5, 6, 7, 8, 9, 10, 11

clc; clear;
%% Load the data from Part 1:
load('Case_Study_part_1.mat', 'data_cell')

%% Extract the data to be used to calibrate the SDF:
data = cell2mat({data_cell{1}, data_cell{3}, data_cell{5}, data_cell{6}});

%% Stochastic model updating parameters:
Nsim = 25; % The number of stochastic model output realisations
N = 1000;  % Define the sample size from posterior
dm = 5;    % To select the JS divergence as the distance function for the ABC procedure

parfor i = 1:4
if i == 1
%% Perform Bayesian model updating via ABC on the p_f of Root event 1:

% Update the hyper-parameters of the Beta distribution:
bounds = [0.01, 100];
priorpdf_R1 = @(x) unifpdf(x(:,1), bounds(1), bounds(2)) .* unifpdf(x(:,2), bounds(1), bounds(2)); 
priorrnd_R1 = @(N) [unifrnd(bounds(1), bounds(2), N, 1), unifrnd(bounds(1), bounds(2), N, 1)]; 
width_par = 0.015; logL_R1 = @(theta) loglikelihood(theta, data(:,1), dm, width_par, Nsim);  

tic;
TEMCMC1 = TEMCMCsampler('nsamples', N, 'loglikelihood', logL_R1, 'priorpdf', priorpdf_R1, 'priorrnd', priorrnd_R1);
timeTEMCMC1 = toc;
sample_R1 = TEMCMC1.samples;

elseif i == 2
%% Perform Bayesian model updating via ABC on the p_f of Root event 3:

% Update the hyper-parameters of the Beta distribution:
bounds = [0.01, 100];
priorpdf_R3 = @(x) unifpdf(x(:,1), bounds(1), bounds(2)) .* unifpdf(x(:,2), bounds(1), bounds(2)); 
priorrnd_R3 = @(N) [unifrnd(bounds(1), bounds(2), N, 1), unifrnd(bounds(1), bounds(2), N, 1)]; 
width_par = 0.015; logL_R3 = @(theta) loglikelihood(theta, data(:,2), dm, width_par, Nsim);  

tic;
TEMCMC2 = TEMCMCsampler('nsamples', N, 'loglikelihood', logL_R3, 'priorpdf', priorpdf_R3, 'priorrnd', priorrnd_R3);
timeTEMCMC2 = toc;
sample_R3 = TEMCMC2.samples;

elseif i == 3
%% Perform Bayesian model updating via ABC on the p_f of Root event 5:

% Update the hyper-parameters of the Beta distribution:
bounds = [0.01, 100];
priorpdf_R5 = @(x) unifpdf(x(:,1), bounds(1), bounds(2)) .* unifpdf(x(:,2), bounds(1), bounds(2)); 
priorrnd_R5 = @(N) [unifrnd(bounds(1), bounds(2), N, 1), unifrnd(bounds(1), bounds(2), N, 1)]; 
width_par = 0.015; logL_R5 = @(theta) loglikelihood(theta, data(:,3), dm, width_par, Nsim);  

tic;
TEMCMC3 = TEMCMCsampler('nsamples', N, 'loglikelihood', logL_R5, 'priorpdf', priorpdf_R5, 'priorrnd', priorrnd_R5);
timeTEMCMC3 = toc;
sample_R5 = TEMCMC3.samples;

elseif i == 4
%% Perform Bayesian model updating via ABC on the p_f of Root event 6:

% Update the hyper-parameters of the Beta distribution:
bounds = [0.01, 100];
priorpdf_R6 = @(x) unifpdf(x(:,1), bounds(1), bounds(2)) .* unifpdf(x(:,2), bounds(1), bounds(2)); 
priorrnd_R6 = @(N) [unifrnd(bounds(1), bounds(2), N, 1), unifrnd(bounds(1), bounds(2), N, 1)]; 
width_par = 0.01; logL_R6 = @(theta) loglikelihood(theta, data(:,4), dm, width_par, Nsim);  

tic;
TEMCMC4 = TEMCMCsampler('nsamples', N, 'loglikelihood', logL_R6, 'priorpdf', priorpdf_R6, 'priorrnd', priorrnd_R6);
timeTEMCMC4 = toc;
sample_R6 = TEMCMC4.samples;

end
end
%% Plot the necessary figures:

% Compile the posterior samples to plot the histograms and Fuzzy-sets:
samples(:,:,1) = sample_R1; samples(:,:,2) = sample_R3; samples(:,:,3) = sample_R5; samples(:,:,4) = sample_R6;

% Plot the posterior histograms:
figure; nbin = 10; f = 20;
label = {'$\alpha_{1}$', '$\alpha_{3}$', '$\alpha_{5}$', '$\alpha_{6}$'; ...
         '$\beta_{1}$', '$\beta_{3}$', '$\beta_{5}$', '$\beta_{6}$'};
for i = 1:size(samples,3)
for j = 1:size(samples,2)
subplot(2, 4, i + ((j-1).*4))
hold on; box on; grid on;
histogram(samples(:,j,i), nbin); xlabel(label{j,i}, 'Interpreter', 'latex'); ylabel('Count', 'FontName', 'Times');
set(gca, 'Fontsize', f); 
end
end

% Plot the posterior-based Fuzzy set:
figure; nbin = 10; f = 20; ci_mat = zeros(2,2,4); % Credible interval matrix
alpha = 0.8; 
true_val = [1.50, 1.50, 1.50, 1.50; 43.00, 43.00, 62.40, 62.40];
for i = 1:size(samples,3)
for j = 1:size(samples,2)
subplot(2, 4, i + ((j-1).*4))
hold on; box on; grid on;
xin = linspace(min(samples(:,j,i)), max(samples(:,j,i)), 1000); yout = makePDF(samples(:,j,i), xin, nbin); 
plot(xin, yout, 'b', 'Linewidth', 2); yline(0.8, 'k--', 'Linewidth', 2); %xline(true_val(j,i), 'r--', 'Linewidth', 2);
xlabel(label{j,i}, 'Interpreter', 'latex'); ylabel('Normalised PDF value', 'FontName', 'Times'); xlim([min(samples(:,j,i)), max(samples(:,j,i))]);
set(gca, 'Fontsize', f)
idx = knnsearch(yout', alpha, 'K', 2); ci_mat(j,:,i) = [xin(sort(idx))];
end
end

%%

% Compute the P-box for the root events:
Ne = 1000;                        % No. of epistemic realizations to generate from the epistemic space
Na = 10000;                       % No. of aleatory realizations from the SDF
pbox = zeros(Na, size(samples,2), size(samples,3));

for i = 1:size(samples,3)
Beta_model = @(x) betarnd(x(:,1), x(:,2), Na, 1); % The SDF RNG model
out_put = DLMC(ci_mat(:,:,i), Beta_model); pbox(:,:,i) = out_put.pbox;
end

% True distribution:
true_dist(:,1) = betarnd(1.50, 43.00, Na, 1);
true_dist(:,2) = betarnd(1.50, 43.00, Na, 1);
true_dist(:,3) = betarnd(1.50, 62.40, Na, 1);
true_dist(:,4) = betarnd(1.50, 62.40, Na, 1);

% Plot the P-box on the P_f of the root events:
figure; f = 25; ylab = {'Probability'};
lab = {'Component 1', 'Component 3', 'Component 5', 'Component 6'};
for i = 1:size(samples,3)
subplot(2,2,i)
hold on; box on; grid on;
[f1,x1] = ecdf(data(:,i)); stairs(x1, f1, 'r', 'linewidth', 2); [f1,x1] = ecdf(true_dist(:,i)); stairs(x1, f1, 'g', 'linewidth', 2);
[f1,x1] = ecdf(pbox(:,1,i)); [f2,x2] = ecdf(pbox(:,2,i));
stairs(x1, f1, 'b', 'linewidth', 2); stairs(x2, f2, 'b', 'linewidth', 2, 'handlevisibility', 'off'); 
plot([min(x1),min(x2)],[0,0], 'b', 'linewidth', 2, 'handlevisibility', 'off'); plot([max(x1),max(x2)],[1,1], 'b', 'linewidth', 2, 'handlevisibility', 'off');
set(gca, 'Fontsize', f); xlabel('$p_{f}$', 'Interpreter', 'latex'); ylabel(ylab, 'FontName', 'Times'); xlim([0, 0.20])
title(lab{i}, 'FontName', 'Times')
end
legend('Data', 'True distribution', 'Probability-box', 'linewidth', 2, 'location', 'southeast')

%% Save data:
save('Case_Study_part_2.mat')
