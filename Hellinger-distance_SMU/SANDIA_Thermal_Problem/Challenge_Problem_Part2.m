%% Part 2: Accreditation Validation:
% 
% For the second part of the challenge, the objective is to perform an 
% accreditation validation on the material thermal property based on the
% accreditation validation experiment data.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load the data from Part 1:
load('Challenge_Problem_Part1_Hellinger.mat', 'TEMCMC1', 'TEMCMC2', 'reg_coeff', 'bounds', 'Nplot', 'nbins', 'k_res', 'v_data')

%% Define the Temperature model:

qin = 3000;        % Heat flux input [W/m^2]
Lin = 0.0190;      % Thickness of the slab [m]

% Define the Temperature model:
% x = Slab thickness coordinates [0, L];
% t = time input [0, 1000];
% k = Thermal conductivity input (Variable);
% v = Volumetric heat capacity (Variable);
T_model = @(x, t, k, v) [Temperature_model(x, t, k, v, qin, Lin)]';

%% Define the accreditation validation experiment data:

t = (0:50:1000)';
exp1 = zeros(length(t),3);
exp1(:, 1) = [25.0, 183.8, 251.3, 302.2, 344.6, 381.7, 414.9, 445.4, 473.6, 500.0, 525.0, 548.8, 571.7, 593.8, 615.2 , 636.1, 656.6, 676.7, 696.4, 716.0, 735.4]';
exp1(:, 2) = [25.0, 26.3, 34.0, 47.7, 64.9, 83.9, 103.7, 124.0, 144.4, 164.9, 185.4, 205.9, 226.3, 246.8, 267.2, 287.6, 307.9, 328.3, 348.6, 369.0, 389.3]';
exp1(:, 3) = [25.0, 25.0, 25.1, 26.0, 28.3, 32.7, 39.3, 48.1, 58.7, 71.1, 84.9, 100.0, 116.1, 133.0, 150.7, 169.0, 187.8, 207.0, 226.5, 246.3, 266.3]';

figure; f = 20; c = {'r', 'g', 'b'};
subplot(2,1,1)
hold on; box on; grid on;
for i = 1:size(exp1,2)
plot(t, exp1(:,i), 'LineWidth', 2, 'color',c{i}, 'handlevisibility', 'off') 
plot(t, exp1(:,i), 's', 'LineWidth', 2, 'MarkerSize', 10, 'MarkerEdgeColor',c{i}, 'MarkerFaceColor', c{i});
end
set(gca, 'Fontsize', f); xlabel('$t$ $[s]$', 'Interpreter', 'latex'); ylabel('$T$ $[^o C]$', 'Interpreter', 'latex'); 
legend('Experiment A (x = 0)', 'Experiment B (x = L/2)', 'Experiment C (x = L)', 'linewidth', 2, 'location', 'northwest')
lab = {'$T(x = 0)$ $[^o C]$', '$T(x = L/2)$ $[^o C]$', '$T(x = L)$ $[^o C]$'}; title_head = {'Experiment A', 'Experiment B', 'Experiment C'};
for i = 1:3
subplot(2,3,i+3)
hold on; box on; grid on;
[f1,x1] = ecdf(exp1(:,i)); stairs(x1, f1, 'color', c{i}, 'linewidth', 2); 
set(gca, 'Fontsize', f); xlabel(lab{i}, 'Interpreter', 'latex'); ylabel('Probability'); title(title_head{i}); 
end

%% Define the key parameter and models:
sample_epsilon_k = TEMCMC1.samples; sample_rho = TEMCMC2.samples; 
xin = [linspace(-0.01, 0.005, Nplot); linspace(0, 0.0002, Nplot); linspace(-3, 2.5, Nplot); linspace(0, 10, Nplot);... 
       linspace(0.036, 0.043, Nplot); linspace(0, 6.2e-05, Nplot); linspace(-2, 2, Nplot); linspace(0, 6.5, Nplot)];

f = zeros(size(sample_epsilon_k,2), Nplot, 2); 
map = zeros(size(sample_epsilon_k,2), 2);      % Maximum A-posteriori values
mean_vec = zeros(size(sample_epsilon_k,2), 2); % Mean values

for i = 1:4
[f(i,:,1), map(i,1)] = makePDF(sample_epsilon_k(:,i), xin(i,:), nbins); [f(i,:,2), map(i,2)] = makePDF(sample_rho(:,i), xin(i+4,:), nbins);
mean_vec(i,:) = mean([sample_epsilon_k(:,i), sample_rho(:,i)]); 
end

N = 1000; % No. of Monte Carlo realizations;
xin = Lin.*[0, 1/2, 1]; delta_threshold = 0.005;

%% Validation exercise 1: Using MAP estimates

t = (0:1:1000)';
SDF_model_epsilon_k_rnd = @(N) SDF_rnd_func(bounds(1,:), [map(1,1), map(1,1); map(2,1), map(2,1); map(3,1), map(3,1); map(4,1), map(4,1)], 2, N, 1);  % The SDF RNG model describing epsilon k
SDF_model_rho_rnd = @(N) SDF_rnd_func(bounds(2,:), [map(1,2), map(1,2); map(2,2), map(2,2); map(3,2), map(3,2); map(4,2), map(4,2)], 2, N, 1).*1e+07; % The SDF RNG model describing rho Cp

T_val_out1 = zeros(length(t), N, 3); % Array for validation output
for i = 1:length(xin)
for j = 1:N
rhoCp_in = SDF_model_rho_rnd(1); eps = SDF_model_epsilon_k_rnd(1);
K_in = @(T) reg_coeff(2,1).*T + reg_coeff(1,1) + eps; % The full linear model for K(T)
T_val_out1(1,j,i) = 25.0;

for k = 2:length(t)
K_in1 = K_in(T_val_out1(k-1,j,i)); Tout1 = T_model(xin(i), t(k), K_in1, rhoCp_in);
K_in2 = K_in(Tout1);               Tout2 = T_model(xin(i), t(k), K_in2, rhoCp_in);
delta = areaMe(Tout1, Tout2);
Tout1 = Tout2;

it = 1; % Initiate convergence loop
while delta > delta_threshold
fprintf('Iteration no. = %d \n', it);
K_in2 = K_in(Tout1); Tout2 = T_model(xin(i), t(k), K_in2, rhoCp_in);
delta = areaMe(Tout1, Tout2);
Tout1 = Tout2; it = it + 1;
end

K_in2 = K_in(Tout1); T_val_out1(k,j,i) = T_model(xin(i), t(k), K_in2, rhoCp_in);

end
end
end

Pbox_T_val1 = zeros(length(t), 2, 3);
for i = 1:3
arr = sort(T_val_out1(:,:,i));
for j = 1:length(t)
Pbox_T_val1(j, :, i) = [min(arr(j, :)), max(arr(j, :))];
end
end

%% Validation exercise 2: Using Mean estimates

t = (0:1:1000)';
SDF_model_epsilon_k_rnd = @(N) SDF_rnd_func(bounds(1,:), [mean_vec(1,1), mean_vec(1,1); mean_vec(2,1), mean_vec(2,1); mean_vec(3,1), mean_vec(3,1); mean_vec(4,1), mean_vec(4,1)], 2, N, 1);  % The SDF RNG model describing epsilon k
SDF_model_rho_rnd = @(N) SDF_rnd_func(bounds(2,:), [mean_vec(1,2), mean_vec(1,2); mean_vec(2,2), mean_vec(2,2); mean_vec(3,2), mean_vec(3,2); mean_vec(4,2), mean_vec(4,2)], 2, N, 1).*1e+07; % The SDF RNG model describing rho Cp

T_val_out2 = zeros(length(t), N, 3); % Array for validation output
for i = 1:length(xin)
for j = 1:N
rhoCp_in = SDF_model_rho_rnd(1); eps = SDF_model_epsilon_k_rnd(1);
K_in = @(T) reg_coeff(2,1).*T + reg_coeff(1,1) + eps; % The full linear model for K(T)
T_val_out2(1,j,i) = 25.0;

for k = 2:length(t)
K_in1 = K_in(T_val_out2(k-1,j,i)); Tout1 = T_model(xin(i), t(k), K_in1, rhoCp_in);
K_in2 = K_in(Tout1);               Tout2 = T_model(xin(i), t(k), K_in2, rhoCp_in);
delta = areaMe(Tout1, Tout2);
Tout1 = Tout2;

it = 1; % Initiate convergence loop
while delta > delta_threshold
fprintf('Iteration no. = %d \n', it);
K_in2 = K_in(Tout1); Tout2 = T_model(xin(i), t(k), K_in2, rhoCp_in);
delta = areaMe(Tout1, Tout2);
Tout1 = Tout2; it = it + 1;
end

K_in2 = K_in(Tout1); T_val_out2(k,j,i) = T_model(xin(i), t(k), K_in2, rhoCp_in);

end
end
end

Pbox_T_val2 = zeros(length(t), 2, 3);
for i = 1:3
arr = sort(T_val_out2(:,:,i));
for j = 1:length(t)
Pbox_T_val2(j, :, i) = [min(arr(j, :)), max(arr(j, :))];
end
end

%% Plot the resulting P-boxes:

figure; f = 20; 
lab = {'$T(x = 0)$ $[^o C]$', '$T(x = L/2)$ $[^o C]$', '$T(x = L)$ $[^o C]$'}; lim_x = [0, 1000; 0, 600; 0, 400];
title_head = {'Experiment A | MAP', 'Experiment B | MAP', 'Experiment C | MAP'; 'Experiment A | Mean', 'Experiment B | Mean', 'Experiment C | Mean'};
for i = 1:3
subplot(2,3,i)
hold on; box on; grid on;
[f1,x1] = ecdf(T_val_out1(:,1,i)); stairs(x1, f1, 'color', [.8 .8 .8], 'linewidth', 2); 
for j = 1:N
[f1,x1] = ecdf(T_val_out1(:,j,i)); stairs(x1, f1, 'color', [.8 .8 .8], 'linewidth', 2, 'handlevisibility', 'off'); 
end
[f1,x1] = ecdf(Pbox_T_val1(:,1,i)); stairs(x1, f1, 'color', 'k', 'linewidth', 2); 
[f1,x1] = ecdf(Pbox_T_val1(:,2,i)); stairs(x1, f1, 'color', 'k', 'linewidth', 2, 'handlevisibility', 'off'); 
[f1,x1] = ecdf(exp1(:,i)); stairs(x1, f1, 'r', 'linewidth', 2); 
set(gca, 'Fontsize', f); xlabel(lab{i}, 'Interpreter', 'latex'); ylabel('Probability'); xlim([lim_x(i,:)]); title(title_head{1,i})

subplot(2,3,i+3)
hold on; box on; grid on;
[f1,x1] = ecdf(T_val_out2(:,1,i)); stairs(x1, f1, 'color', [.8 .8 .8], 'linewidth', 2); 
for j = 1:N
[f1,x1] = ecdf(T_val_out2(:,j,i)); stairs(x1, f1, 'color', [.8 .8 .8], 'linewidth', 2, 'handlevisibility', 'off'); 
end
[f1,x1] = ecdf(Pbox_T_val2(:,1,i)); stairs(x1, f1, 'color', 'k', 'linewidth', 2); 
[f1,x1] = ecdf(Pbox_T_val2(:,2,i)); stairs(x1, f1, 'color', 'k', 'linewidth', 2, 'handlevisibility', 'off'); 
[f1,x1] = ecdf(exp1(:,i)); stairs(x1, f1, 'r', 'linewidth', 2); 
set(gca, 'Fontsize', f); xlabel(lab{i}, 'Interpreter', 'latex'); ylabel('Probability'); xlim([lim_x(i,:)]); title(title_head{2,i})
end
legend('Model prediction', 'P-box on Model prediction', 'Validation data', 'linewidth', 2, 'location', 'southeast')

%% Obtain validation statistics:

stats_vec = zeros(2,3,2);

for i = 1:size(exp1,2)
area1 = zeros(N,1); area2 = zeros(N,1);
for j = 1:N
area1(j) = areaMe(T_val_out1(:,j,i), exp1(:,i));
area2(j) = areaMe(T_val_out2(:,j,i), exp1(:,i));
end
stats_vec(:,i,1) = [mean(area1), std(area1)]'; stats_vec(:,i,2) = [mean(area2), std(area2)]';  
end

beta_vec = [stats_vec(1,1,1), stats_vec(1,1,2) ; stats_vec(1,2,1), stats_vec(1,2,2) ; stats_vec(1,3,1), stats_vec(1,3,2)];
err = [stats_vec(2,1,1), stats_vec(2,1,2) ; stats_vec(2,2,1), stats_vec(2,2,2) ; stats_vec(2,3,1), stats_vec(2,3,2)];
iterations = [1, 2, 3];

figure; f = 20;
hold on; box on; grid on;
b = bar(iterations, beta_vec, 'linewidth', 2);

% Calculate the number of groups and number of bars in each group
[ngroups,nbars] = size(beta_vec);
% Get the x coordinate of the bars
x = nan(nbars, ngroups);
for i = 1:nbars
x(i,:) = b(i).XEndPoints;
end

% Plot the errorbars
errorbar(x', beta_vec, err, 'k', 'linestyle', 'none', 'linewidth', 2);
xlabel('Validation experiment no.'); ylabel('Area metric value $[^o C]$', 'Interpreter', 'latex'); xticks([1:3])
set(gca, 'Fontsize', f); legend('Model validation (Posterior MAP)', 'Model validation (Posterior Mean)', 'linewidth', 2)

%% Comparing SDF distributions with ECDF of data:

Na = 10000;

figure; ylab = {'Probability'}; f = 20;
subplot(1,2,1)
hold on; box on; grid on;
[f1,x1] = ecdf(k_res); stairs(x1, f1, 'r', 'linewidth', 2); 
SDF_model_epsilon_k_rnd = @(N) SDF_rnd_func(bounds(1,:), [map(1,1), map(1,1); map(2,1), map(2,1); map(3,1), map(3,1); map(4,1), map(4,1)], 2, N, 1);  % The SDF RNG model describing epsilon k
epsilon_k_1 = SDF_model_epsilon_k_rnd(Na); [f1,x1] = ecdf(epsilon_k_1); stairs(x1, f1, 'b', 'linewidth', 2);
SDF_model_epsilon_k_rnd = @(N) SDF_rnd_func(bounds(1,:), [mean_vec(1,1), mean_vec(1,1); mean_vec(2,1), mean_vec(2,1); mean_vec(3,1), mean_vec(3,1); mean_vec(4,1), mean_vec(4,1)], 2, N, 1);  % The SDF RNG model describing epsilon k
epsilon_k_2 = SDF_model_epsilon_k_rnd(Na); [f1,x1] = ecdf(epsilon_k_2); stairs(x1, f1, 'g', 'linewidth', 2);
set(gca, 'Fontsize', f); xlabel('$\epsilon_k$ $[W/{m} \cdot {^oC}]$', 'Interpreter', 'latex'); ylabel(ylab); 

subplot(1,2,2)
hold on; box on; grid on;
[f1,x1] = ecdf(v_data(:)); stairs(x1, f1, 'r', 'linewidth', 2); 
SDF_model_rho_rnd = @(N) SDF_rnd_func(bounds(2,:), [map(1,2), map(1,2); map(2,2), map(2,2); map(3,2), map(3,2); map(4,2), map(4,2)], 2, N, 1).*1e+07; % The SDF RNG model describing rho Cp
rhoCp_1 = SDF_model_rho_rnd(Na); [f1,x1] = ecdf(rhoCp_1); stairs(x1, f1, 'b', 'linewidth', 2);
SDF_model_rho_rnd = @(N) SDF_rnd_func(bounds(2,:), [mean_vec(1,2), mean_vec(1,2); mean_vec(2,2), mean_vec(2,2); mean_vec(3,2), mean_vec(3,2); mean_vec(4,2), mean_vec(4,2)], 2, N, 1).*1e+07; % The SDF RNG model describing rho Cp
rhoCp_2 = SDF_model_rho_rnd(Na); [f1,x1] = ecdf(rhoCp_2); stairs(x1, f1, 'g', 'linewidth', 2);
set(gca, 'Fontsize', f); xlabel('$\rho C_p$ $[J/{m^3} \cdot {^oC}]$', 'Interpreter', 'latex'); ylabel(ylab); 
legend('Experiment', 'Maximum A-posteriori', 'Mean', 'linewidth', 2, 'location', 'southeast')

%% Save the data:
save('Challenge_Problem_Part2_Hellinger')