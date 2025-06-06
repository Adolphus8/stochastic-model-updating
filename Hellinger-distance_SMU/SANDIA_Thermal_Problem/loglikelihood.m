function [logL] = loglikelihood(theta, data, bounds, width, Nsim)
%% The Log-likelihood function defined for the Bayesian model updating procedure:
%
% Inputs:
% theta:  the N x dim matrix of input samples;
% data:   the Nobs x 1 vector of training data;
% bounds: the 1 x 2 vector of bounds on the Staircase Density Function;
% width:  the scalar value of the width parameter of the loglikelihood function;
% Nsim:   the scalar number of simulated realisations of the stochastic model output;
%
% Output:
% logL:  the N x 1 vector of loglikelihood values;
%% Define the parameters:
N = size(theta,1); % The input sample size

%% Computation procedure:

logL = zeros(N,1); 
for i = 1:N

x_sim = SDF_rnd(theta, bounds, 2, Nsim); dist = Hellinger(x_sim, data);
logL(i) = - (dist./width).^2;

if isinf(logL(i)) || isnan(logL(i))
logL(i) = -1e+100;    
end
end

end

function [output] = Hellinger(samp1, samp2)
% Return the Hellinger distance between y_sim and y_exp based on the adaptive binning algorithm
%%
% INPUT: 
% samp1 (model_samp): Nsim x dim matrix of simulated model output y_sim samples;
% samp2 (data_samp):  Nobs x dim matrix of y_obs samples;
% 
% OUTPUT: 
% output:     The scalar value of the Hellinger distance;
%%
% Compute the Bhattacharyya distance:
BD_value = BDMe(samp1, samp2);

% Compute the Bhattacharyya coefficient:
BC = exp(- BD_value);

% Compute the Hellinger distance:
output = sqrt(1 - BC);

end

function bd = BDMe(sample_1, sample_2)
% Return the Bhattacharrya distance between y_sim and y_exp based on the binning algorithm
%%
% INPUT: 
% sample_1: Nsim x dim matrix of simulated model output y_sim samples;
% sample_2: Nobs x dim matrix of data y_exp samples;
% Nbin:     Scalar number of bins
%
% OUTPUT: 
% bd:       The scalar value of the Bhattacharrya distance;
%%
% Define the variables:
[Nsamp1, dim1] = size(sample_1); [Nsamp2, dim2] = size(sample_2);

if dim1 ~= dim2
error('No. of column(s) of the two samples must be equal to each other.')    
end

%% Initiate the Adaptive-binning algorithm:

EDme = EDMe(sample_1, sample_2); % Compute the Euclidean distance

delta_vec = zeros(dim2,1);
for i = 1:dim2
delta_mat = zeros(Nsamp2, Nsamp2);    
for j = 1:Nsamp2
for k = 1:Nsamp2
delta_mat(j,k) = abs(sample_1(j,i) - sample_2(k,i));
end
end
delta_vec(i,1) = max(delta_mat, [], 'all');
end

delta_sim = max(delta_vec);
bin_width = (log(delta_sim + 1)./max([Nsamp1^(1/3), Nsamp2^(1/3)])) .* exp(EDme); % Compute the bin width
Nbin = ceil(delta_sim./bin_width); % Compute number of bins

if Nbin < 2
Nbin = 2;
elseif Nbin > ceil(max(Nsamp1, Nsamp2)./10)
Nbin = ceil(max(Nsamp1, Nsamp2)./10);
else
Nbin = Nbin;
end

%% Compute the PMF function for each sample:

max_1 = max(sample_1); min_1 = min(sample_1);
max_2 = max(sample_2); min_2 = min(sample_2);

ub = max([max_1; max_2]); lb = min([min_1; min_2]);

% This treatment is necessary to avoid unexpected error in the following frequenty counting process:
ub = ub + abs(ub*0.0001); lb = lb - abs(lb*0.0001); 

intervals = zeros(Nbin + 1, dim1);
for icolumn = 1:dim1
intervals(:, icolumn) = linspace(lb(icolumn), ub(icolumn), Nbin + 1);
end

% Compute the histogram counts:
count_1 = histcounts(sample_1, intervals); count_2 = histcounts(sample_2, intervals);
count_ratio_1 = count_1/Nsamp1; count_ratio_2 = count_2/Nsamp2;

% Compute the distance metric:
dis = zeros(Nbin, 1);
m = 1;
for i = 1:Nbin
dis(m) = sqrt(count_ratio_1(i)*count_ratio_2(i));
m = m + 1;
end

bd = -log(sum(dis));
end

function ed = EDMe(sample_1,sample_2)
% Return the Euclidean distance between y_sim and y_exp
%%
% INPUT: 
% sample_1: N x dim matrix of simulated model output y_sim samples;
% sample_2: N x dim matrix of data y_exp samples;
%
% OUTPUT: 
% ed:       The scalar value of the Euclidean distance;
%%
% Define the variables:
dim1 = size(sample_1,2); dim2 = size(sample_2,2);

if dim1 ~= dim2
error('No. of column(s) of the two samples must be equal to each other.')    
end

mean_diff_squared = zeros(dim1,1);
for i = 1:dim1
mean_diff_squared(i) = (mean(sample_1(:,i)) - mean(sample_2(:,i))).^2;    
end

ed = sqrt(sum(mean_diff_squared));
end

function [samples] = SDF_rnd(theta, bounds, objective, Nsamp)
% This is the function handle of the Staircase Density Function Random Number Generator:

% Inputs:
% theta:     N x dim vector of epistemic Staricase Density Function parameters;
% bounds:    Bounds of the aleatory parameters;
% objective: Numerical objective function flag for the optimization problem;
% Nsamp:     Numerical value of the number of samples to obtain from the joint distribution defined by the Staircase Density Function;

% Output:
% samples:   The Nsamp x dim vector of sample output;

%% Error check:
assert(size(theta,1)==1);

%% Define the function:
objective_func = objective;
N = Nsamp; 
dim = size(theta,2)./4;

theta_a = cell(1);
samples = zeros(N, dim);
for ia = 1:dim
theta_a{ia} = theta(1 + 4*(ia - 1):4*ia);
    
% Fit a staircase density:
[z_i, l, ~] = staircasefit(bounds, theta_a{ia}, objective_func);
l(l < 0) = 0;
    
% Generate samples from the staircase density:
samples(:, ia) = staircasernd(N, z_i, l, bounds);
end

end

function [z_i, l, c_i] = staircasefit(bounds, theta, objective, varargin)
% Calculation of the staircase random variables
%
%     INPUT : bounds    -- prior distribution of aleatory parameters
%             theta     -- samples of the epistemic parameters
%             objective -- objective function flag for the optimization problem
%
%     OUTPUT : z_i -- partitioning points
%              l   -- staircase heights
%              c_i -- centers of the bins
%

n_b = 50;   % n. of bins of staircase RVs
if nargin >= 4
    n_b = varargin{1};
end

z_i = linspace(bounds(1), bounds(2), n_b + 1);   % partitioning points
kappa = diff(bounds)/n_b;                        % subintervals
c_i = z_i(1:end - 1) + kappa/2;                  % centers of the bins

[feasible, ~] = isfeasible(bounds, theta);

theta(3) = theta(3)*theta(2)^(3/2);
theta(4) = theta(4)*theta(2)^2;

if feasible
    Aeq = [kappa*ones(size(c_i));
        kappa*c_i;
        kappa*c_i.^2 + kappa^3/12;
        kappa*c_i.^3 + kappa^3*c_i/4;
        kappa*c_i.^4 + kappa^3*c_i.^2/2 + kappa^5/80];
    beq = [1;
        theta(1);
        theta(1)^2 + theta(2);
        theta(3)+3*theta(1)*theta(2) + theta(1)^3;
        theta(4) + 4*theta(3)*theta(1) + 6*theta(2)*theta(1)^2 + theta(1)^4];
    
    options = optimoptions('fmincon', 'Display', 'off', 'Algorithm', 'sqp');
    options.MaxFunctionEvaluations = 10000*n_b;
    options.MaxIterations = 1000;
    options.ConstraintTolerance = 1e-6;
    options.StepTolerance = 1e-12;
    switch objective
        case 1
            J = @(l) kappa*log(l)*l';
        case 2
            J = @(l) l*l';
        case 3
            J = @(l) -omega*log(l)';
    end
    
    % Do fmincon starting from a uniform distribution over bounds
    l = fmincon(J, 1/(diff(bounds))*ones(size(c_i)), [], [], Aeq, beq, zeros(size(c_i)), [], [], options);
else
    error('unfeasible set of parameters')
end

l(l < 0) = 0;   % ignore negative values

end

function [Lfeasible, constraints] = isfeasible(bounds, theta)
% Return a column of the prior pdf.
%
%     INPUT : bouds -- prior distribution of aleatory parameters
%             theta -- theta(:, 1): mean
%                      theta(:, 2): variance
%                      theta(:, 3): the third-order central moment
%                      theta(:, 4): the fourth-order central moment
%
%     OUTPUT : Lfeasible   -- the prior pdf
%              constraints -- the moment constraints
%

Nsample = size(theta, 1);
Lfeasible = zeros(Nsample, 1);

theta(:, 3) = theta(:, 3).*theta(:, 2).^(3/2);
theta(:, 4) = theta(:, 4).*theta(:, 2).^2;

for isample = 1:Nsample
    u = bounds(1) + bounds(2) - 2*(theta(isample, 1));
    v = (theta(isample, 1) - bounds(1))*(bounds(2) - theta(isample, 1));
    
    constraints = [bounds(1) - theta(isample, 1);   % g2
        theta(isample, 1) - bounds(2);   % g3
        -theta(isample, 2);   % g4
        theta(isample, 2) - v;   % g5
        theta(isample, 2)^2 - theta(isample, 2)*(theta(isample, 1) - bounds(1))^2 - theta(isample, 3)*...
        (theta(isample, 1) - bounds(1));   % g6
        theta(isample, 3)*(bounds(2) - theta(isample, 1)) - theta(isample, 2)*...
        (bounds(2) - theta(isample, 1))^2 + theta(isample, 2)^2;   % g7
        4*theta(isample, 2)^3 + theta(isample, 3)^2 - theta(isample, 2)^2*diff(bounds)^2;   % g8
        6*sqrt(3)*theta(isample, 3) - diff(bounds)^3;   % g9
        -6*sqrt(3)*theta(isample, 3) - diff(bounds)^3;   % g10
        -theta(isample, 4);   % g11
        12*theta(isample, 4) - diff(bounds)^4;   % g12
        (theta(isample, 4) - v*theta(isample, 2) - u*theta(isample, 3))*...
        (v - theta(isample, 2)) + (theta(isample, 3)- u*theta(isample, 2))^2;   % g13
        theta(isample, 3)^2 + theta(isample, 2)^3 - theta(isample, 4)*theta(isample, 2)];   % g14
    
    Lfeasible(isample) = all(constraints <= 0);
end
end

function x = staircasernd(N, z_i, l, bounds)
% Return a column of parameters sampled from the prior pdf
%
%     INPUT : N       -- n. of samples
%             z_i     -- partitioning points
%             l       -- staircase heights
%             bounds  -- prior distribution of aleatory parameters
%
%     OUTPUT : x -- matrix of samples from x_pdf
%

n_b = length(l);   % n. of bins of staircase RVs
idx = (randsample(length(l), N, true, diff(bounds)/n_b*l));   % select random stair
x = unifrnd(z_i(idx), z_i(idx + 1))';   % generate uniform in each stair
end
