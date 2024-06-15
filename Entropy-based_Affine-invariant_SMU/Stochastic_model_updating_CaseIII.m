%% Material characterisation - Stochastic model updating:
% 
% For the first part of the challenge, the objective is to characterise the
% variability of the thermal properties of the given specimen based on a
% given limited data set.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Define the data set:

temperature_data = [20, 250, 500, 750, 1000];

k_data = [0.0496, 0.0628, 0.0602, 0.0657, 0.0631; 0.0530, 0.0620, 0.0546, 0.0713, 0.0796; ...
          0.0493, 0.0537, 0.0638, 0.0694, 0.0692; 0.0455, 0.0561, 0.0614, 0.0732, 0.0739; ...
          0.0483, 0.0563, 0.0643, 0.0684, 0.0806; 0.0490, 0.0622, 0.0714, 0.0662, 0.0811];

v_data = [3.76E+05, 3.87E+05, 4.52E+05, 4.68E+05, 4.19E+05; 3.38E+05, 4.69E+05, 4.10E+05, 4.24E+05, 4.38E+05; ...
          3.50E+05, 4.19E+05, 4.02E+05, 3.72E+05, 3.45E+05; 4.13E+05, 4.28E+05, 3.94E+05, 3.46E+05, 3.95E+05; ...
          4.02E+05, 3.37E+05, 3.73E+05, 4.07E+05, 3.78E+05; 3.53E+05, 3.77E+05, 3.69E+05, 3.99E+05, 3.77E+05];

%% Pair the data to compute the Pearson correlation:
clc;

k_temp = [repmat(temperature_data, [1, size(k_data,1)]); [k_data(1,:), k_data(2,:), k_data(3,:), k_data(4,:), k_data(5,:), k_data(6,:)]]'; k_temp_corr = corrcoef(k_temp);
sprintf('The Pearson correlation coefficient between Temperature and Thermal conductivity is = %0.5g', k_temp_corr(1,2))

v_temp = [repmat(temperature_data, [1, size(v_data,1)]); [v_data(1,:), v_data(2,:), v_data(3,:), v_data(4,:), v_data(5,:), v_data(6,:)]]'; v_temp_corr = corrcoef(v_temp);
sprintf('The Pearson correlation coefficient between Temperature and Volumetric heat capacity is = %0.5g', v_temp_corr(1,2))

%% Illustrating relationship between Thermal conductivity / Volumetric heat capacity vs Temperature:
% From the results, it is observed that the Pearson correlation between Temperature and Thermal conductivity is very high:
% A linear regression can be performed for Thermal conductivity vs Temperature.

tbl = table(k_temp(:,1), k_temp(:,2), 'VariableNames', {'Temperature','Thermal_conductivity'});
lm = fitlm(tbl,'linear');
reg_coeff = table2array(lm.Coefficients);
sigma_k = sqrt(lm.MSE); % Root Mean Squared Error

% Nominal Temperature-dependent model of the Thermal conductivity (without accounting for the residual error):
K_model = @(T) reg_coeff(2,1).*T + reg_coeff(1,1);

% Compute the residual of k:
m = reg_coeff(2,1);
k_res = [k_data(:,1)' - ((m .* temperature_data(1)) + reg_coeff(1,1)),...
         k_data(:,2)' - ((m .* temperature_data(2)) + reg_coeff(1,1)), ...
         k_data(:,3)' - ((m .* temperature_data(3)) + reg_coeff(1,1)), ...
         k_data(:,4)' - ((m .* temperature_data(4)) + reg_coeff(1,1)), ...
         k_data(:,5)' - ((m .* temperature_data(5)) + reg_coeff(1,1))]'; 

figure;
subplot(2,2,1)
hold on; box on; grid on;
for i = 1:size(k_data,1)
scatter(temperature_data, k_data(i,:), 30, 'blue', 'filled', 'HandleVisibility','off')
end
plot([0,temperature_data], [reg_coeff(2,1).*[0, temperature_data] + reg_coeff(1,1)], 'r', 'linewidth', 1.5)
legend('Line of best-fit', 'linewidth', 2, 'location', 'southeast')
xlabel('$T$ $[^oC]$', 'Interpreter', 'latex'); ylabel('$k$ $[W/{m} \cdot {^oC}]$', 'Interpreter', 'latex'); set(gca, 'Fontsize', 20)
subplot(2,2,2)
hold on; box on; grid on;
for i = 1:size(v_data,1)
scatter(temperature_data, v_data(i,:), 30, 'blue', 'filled')
end
xlabel('$T$ $[^oC]$', 'Interpreter', 'latex'); ylabel('$\rho$ $C_p$ $[J/{m^3} \cdot {^oC}]$', 'Interpreter', 'latex'); set(gca, 'Fontsize', 20)
subplot(2,2,3)
hold on; box on; grid on;
histogram(k_res, 5);
xlabel('$\epsilon_k$ $[W/{m} \cdot {^oC}]$', 'Interpreter', 'latex'); ylabel('Count'); set(gca, 'Fontsize', 20)
subplot(2,2,4)
hold on; box on; grid on;
histogram(v_data, 5);
xlabel('$\rho$ $C_p$ $[J/{m^3} \cdot {^oC}]$', 'Interpreter', 'latex'); ylabel('Count'); set(gca, 'Fontsize', 20)

%% Perform Bayesian model updating via ABC on the distribution of the Thermal conductivity & Volumetric heat capacity:

% To loosen any assumption on the distribution model, we use the Staircase Density Function to model the aleatory variability.

% Priors for the Thermal conductivity & Volumetric heat capacity:
bounds = [-0.02, 0.02; 0.03, 0.05];
Nsim = 40;                                    % The number of stochastic model output realisations
N = 1000;                                     % Define the sample size from posterior
dm = 5;                                       % dm is the index for the type of distance metric

% Update the hyper-parameters of the SDF for the Thermal conductivity residual variable (epsilon_k):
priorpdf_epsilon_k = @(x) isfeasible(bounds(1,:), x);  % Prior for Thermal conductivity hyper parameters;
priorrnd_epsilon_k = @(N) prior_staircaseRV(N, 1, bounds(1,:)); 
width_k = 0.05;
logL_k = @(theta) loglikelihood(theta, k_res, bounds(1,:), dm, width_k, Nsim);  

tic;
TEMCMC1 = TEMCMCsampler('nsamples', N, 'loglikelihood', logL_k, 'priorpdf', priorpdf_epsilon_k, 'priorrnd', priorrnd_epsilon_k);
timeTEMCMC1 = toc;
sample_epsilon_k = TEMCMC1.samples;

% Update the hyper-parameters of the SDF for the Volumetric heat capacity variable (rhoCp):
priorpdf_rho = @(x) isfeasible(bounds(2,:), x);  % Prior for Volumetric heat capacity hyper parameters;
priorrnd_rho = @(N) prior_staircaseRV(N, 1, bounds(2,:)); 
width_rho = 0.012;
logL_rho = @(theta) loglikelihood(theta, v_data(:).*1e-07, bounds(2,:), dm, width_rho, Nsim); 

tic;
TEMCMC2 = TEMCMCsampler('nsamples', N, 'loglikelihood', logL_rho, 'priorpdf', priorpdf_rho, 'priorrnd', priorrnd_rho);
timeTEMCMC2 = toc;
sample_rho = TEMCMC2.samples; 
sample_rho(:,1) = sample_rho(:,1) .* 1e+07; sample_rho(:,2) = sample_rho(:,2) .* (1e+07).^2; % Re-scale the mean and variance

figure; nbins = 10; fz = 15;
label = {'$\mu_{\epsilon_k}$ $[W/{m} \cdot {^oC}]$', '$({m_2})_{\epsilon_k}$ $[(W/{m} \cdot {^oC})^2]$', '$({m_3}/{m_2}^{3/2})_{\epsilon_k}$', '$({m_4}/{m_2}^{2})_{\epsilon_k}$'; ...
         '$\mu_{\rho C_p}$ $[J/{m^3} \cdot {^oC}]$', '$({m_2})_{\rho C_p}$ $[(J/{m^3} \cdot {^oC})^2]$', '$({m_3}/{m_2}^{3/2})_{\rho C_p}$', '$({m_4}/{m_2}^{2})_{\rho C_p}$'};
for i = 1:4
subplot(2,4,i)
hold on; box on; grid on;
histogram(sample_epsilon_k(:,i), nbins)
set(gca, 'Fontsize', fz); xlabel(label{1,i}, 'Interpreter', 'latex'); ylabel('Count')

subplot(2,4,4+i)
hold on; box on; grid on;
histogram(sample_rho(:,i), nbins)
set(gca, 'Fontsize', fz); xlabel(label{2,i}, 'Interpreter', 'latex'); ylabel('Count')
end

%% Compute the credible intervals:
a = 43; b = 43;
ci_epsilon_k = [prctile(TEMCMC1.samples, [a, 100-a])]'; ci_rho = [prctile(TEMCMC2.samples, [b, 100-b])]';

%% Double-loop Monte Carlo step:

% Define the parameters and model for Double-Loop Monte Carlo:
Ne = 1000;                        % No. of epistemic realizations to generate from the epistemic space
Na = 10000;                       % No. of aleatory realizations from the stochastic black-box model

SDF_model_epsilon_k = @(alpha_cut) SDF_rnd_func(bounds(1,:), alpha_cut, 2, Na, Ne); % The SDF RNG model describing epsilon k
out_epsilon_k = DLMC([prctile(TEMCMC1.samples, [a, 100-a])]', SDF_model_epsilon_k);

SDF_model_rho = @(alpha_cut) SDF_rnd_func(bounds(2,:), alpha_cut, 2, Na, Ne);       % The SDF RNG model describing rho
out_rho = DLMC([prctile(TEMCMC2.samples, [b, 100-b])]', SDF_model_rho);

%% Construct the Probability boxes (P-boxes):

pbox_epsilon_k = out_epsilon_k.pbox; pbox_rho = (out_rho.pbox).*1e+07;

figure; ylab = {'Probability'}; f = 18;
subplot(1,2,1)
hold on; box on; grid on;
[f1,x1] = ecdf(k_res); stairs(x1, f1, 'r', 'linewidth', 2); 
[f1,x1] = ecdf(pbox_epsilon_k(:,1)); [f2,x2] = ecdf(pbox_epsilon_k(:,2));
stairs(x1, f1, 'b', 'linewidth', 2); stairs(x2, f2, 'b', 'linewidth', 2, 'handlevisibility', 'off'); 
plot([min(x1),min(x2)],[0,0], 'b', 'linewidth', 2, 'handlevisibility', 'off'); 
plot([max(x1),max(x2)],[1,1], 'b', 'linewidth', 2, 'handlevisibility', 'off');
set(gca, 'Fontsize', f); xlabel('$\epsilon_k$ $[W/{m} \cdot {^oC}]$', 'Interpreter', 'latex'); ylabel(ylab); 

subplot(1,2,2)
hold on; box on; grid on;
[f1,x1] = ecdf(v_data(:)); stairs(x1, f1, 'r', 'linewidth', 2); 
[f1,x1] = ecdf(pbox_rho(:,1)); [f2,x2] = ecdf(pbox_rho(:,2));
stairs(x1, f1, 'b', 'linewidth', 2); stairs(x2, f2, 'b', 'linewidth', 2, 'handlevisibility', 'off'); 
plot([min(x1),min(x2)],[0,0], 'b', 'linewidth', 2, 'handlevisibility', 'off'); 
plot([max(x1),max(x2)],[1,1], 'b', 'linewidth', 2, 'handlevisibility', 'off');
set(gca, 'Fontsize', f); xlabel('$\rho C_p$ $[J/{m^3} \cdot {^oC}]$', 'Interpreter', 'latex'); ylabel(ylab); 

%% Save the data:
save('Stochastic_model_updating_CaseIII')
