%% The Physics-guided Reliability Analysis:
% 
% For the second part of the challenge, the objective is to perform the 
% reliability analysis (under independence between epsilon_k and rhoCp) of 
% the slab material accounting for the variability in the thermal properties 
% of the given specimen.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load the data from the Stochastic model updating step:
cse = 3;

if cse == 1
load('Stochastic_model_updating_CaseI.mat')
elseif cse == 2
load('Stochastic_model_updating_CaseII.mat')
elseif cse == 3
load('Stochastic_model_updating_CaseIII.mat')
end

%% Define the Temperature model and Performance function for the reliability analysis:

xin = 0;           % Surface coordinates [m]
tin = 1000;        % Threshold time [s]
qin = 3500;        % Heat flux input [W/m^2]
Lin = 0.0190;      % Thickness of the slab [m]
T_threshold = 900; % Threshold temperature [deg]

% Define the Temperature model:
T_model = @(k,v) [Temperature_model(xin,tin,k,v,qin,Lin)]';

% Define the performance function:
% Safe domain: g < 0
% Fail domain: g > 0
g_func = @(k,v) T_model(k,v) - T_threshold;

%% Perform the Reliability analysis:

Na = 10000; Ne = 1000;
samp_epsilon_k = @(m) SDF_rnd_func(bounds(1,:), [m;m]', 2, Na, 1);    % Function-handle to obtain the Na x 1 realisations of epsilon k
samp_rho = @(m) (SDF_rnd_func(bounds(2,:), [m;m]', 2, Na, 1)).*1e+07; % Function-handle to obtain the Na x 1 realisations of rho

threshold = 0.6;      % Define the convergence criteria
g_out = zeros(Na,Ne); % Empty array for the performance function values

epsilon_k_moments = zeros(Ne, 4); rho_moments = zeros(Ne, 4); 

tic;
for i = 1:Ne % Start of outer loop to compute performance function value (For each epistemic realization)
clc; fprintf('Outer loop iteration no. = %d \n', i);

epsilon_k_moments(i,:) = [unifrnd(ci_epsilon_k(1,1), ci_epsilon_k(1,2), 1, 1), unifrnd(ci_epsilon_k(2,1), ci_epsilon_k(2,2), 1, 1), ...
                          unifrnd(ci_epsilon_k(3,1), ci_epsilon_k(3,2), 1, 1), unifrnd(ci_epsilon_k(4,1), ci_epsilon_k(4,2), 1, 1)];

rho_moments(i,:) = [unifrnd(ci_rho(1,1), ci_rho(1,2), 1, 1), unifrnd(ci_rho(2,1), ci_rho(2,2), 1, 1), ...
                    unifrnd(ci_rho(3,1), ci_rho(3,2), 1, 1), unifrnd(ci_rho(4,1), ci_rho(4,2), 1, 1)];

K_model_noisy = @(T,m) (reg_coeff(2,1).*T + reg_coeff(1,1)) + samp_epsilon_k(m); % Generate noisy values of k given temperature T

% Generate the starting seed samples from a Normal distribution - At this point introduce the dependency between k and rhoCp 
K_input = normrnd(mean(k_data(:)), std(k_data(:)), Na, 1); 

T_dist = zeros(Na,1); rho_samp = samp_rho(rho_moments(i,:));
for j = 1:Na
T_dist(j) = T_model(K_input(j), rho_samp(j));
end

K_output = K_model_noisy(T_dist, epsilon_k_moments(i,:));

T_dist_new = zeros(Na,1); rho_samp = samp_rho(rho_moments(i,:));
for j = 1:Na
T_dist_new(j) = T_model(K_output(j), rho_samp(j));
end

area = areaMe(T_dist_new, T_dist); T_dist = T_dist_new;

% Start of iterative loop to converge the k(T) distribution
it = 1;
while area > threshold
fprintf('Iteration no. = %d \n', it);

K_input = K_model_noisy(T_dist, epsilon_k_moments(i,:));

T_dist_new = zeros(Na,1); rho_samp = samp_rho(rho_moments(i,:));
for j = 1:Na
T_dist_new(j) = T_model(K_input(j), rho_samp(j));
end

area = areaMe(T_dist_new, T_dist);

it = it + 1; T_dist = T_dist_new;
end
% End of iterative loop to converge the k(T) distribution

% Compute the performance function value:
K_input = K_model_noisy(T_dist, epsilon_k_moments(i,:)); rho_samp = samp_rho(rho_moments(i,:));
for j = 1:Na
g_out(j,i) = g_func(K_input(j), rho_samp(j)); 
end
end % End of computation loop of the performance function
timeDLMC = toc;

%% Plot the graphs:

g_output = sort(g_out);
g_pbox = zeros(Na,2);
for i = 1:Na
g_pbox(i,:) = [min(g_output(i,:)), max(g_output(i,:))];    
end

% Plot ECDF curve of g:
figure; ylab = {'Probability'}; f = 18;
subplot(2,1,1)
hold on; box on; grid on;
[f1,x1] = ecdf(g_pbox(:,1)); stairs(x1, f1, 'b', 'linewidth', 2); 
[f2,x2] = ecdf(g_pbox(:,2)); stairs(x2, f2, 'b', 'linewidth', 2, 'handlevisibility', 'off');
plot([min(x1),min(x2)],[0,0], 'b', 'linewidth', 2, 'handlevisibility', 'off'); 
plot([max(x1),max(x2)],[1,1], 'b', 'linewidth', 2, 'handlevisibility', 'off');
xline(0, 'k --', 'linewidth', 2)
set(gca, 'Fontsize', f); xlabel('$g$ $[^o C]$', 'Interpreter', 'latex'); ylabel(ylab); 

prob = zeros(Ne,1);
for i = 1:Ne
idx = find(g_output(:,i) > 0);
prob(i) = length(idx)./Na;
end

% Plot histogram curve of P(g > 0):
subplot(2,2,3); ylab = {'Count'}; 
hold on; box on; grid on;
histogram(prob, 10, 'FaceColor','m')
xline(0.01, 'k --', 'linewidth', 2)
set(gca, 'Fontsize', f); xlabel('$P(g > 0^o C)$', 'Interpreter', 'latex'); ylabel(ylab); 

% Plot ECDF curve of P(g > 0):
subplot(2,2,4); ylab = {'Probability'}; 
hold on; box on; grid on;
[f1,x1] = ecdf(prob); stairs(x1, f1, 'm', 'linewidth', 2); 
xline(0.01, 'k --', 'linewidth', 2)
set(gca, 'Fontsize', f); xlabel('$P(g > 0^o C)$', 'Interpreter', 'latex'); ylabel(ylab); 

%% Report the final statistics:
clc;

% Report the failure interval of P(g > 0):
failure = ([length(find(g_pbox(:,1) > 0)), length(find(g_pbox(:,2) > 0))]./Na);
[fprintf('The failure interval is: p_f = ['); fprintf('%g, ', failure(1)); fprintf('%g]\n', failure(2))];

% Report the probability that the regulatory requirement that p(T_s(t=t') > T_f) < 0.01 is met across Ne runs:
p_rel = length(find(prob <= 0.01))./length(prob);
fprintf('The probability of meeting the regulatory requirement is: p_rel = %g\n', p_rel)

%% Save the data:

if cse == 1
save('Reliability_analysis_CaseI_Independence', 'g_out', 'g_pbox', 'epsilon_k_moments', 'rho_moments', 'timeDLMC', 'failure', 'p_rel') 
elseif cse == 2
save('Reliability_analysis_CaseII_Independence', 'g_out', 'g_pbox', 'epsilon_k_moments', 'rho_moments', 'timeDLMC', 'failure', 'p_rel')
elseif cse == 3
save('Reliability_analysis_CaseIII_Independence', 'g_out', 'g_pbox', 'epsilon_k_moments', 'rho_moments', 'timeDLMC', 'failure', 'p_rel')
end
