function [output] = DLMC(bounds, model)
%% The Double-Loop Monte Carlo function:
%
% Inputs:
% bounds: bounds on the SDF;
% Ne:     the scalar value of the number of epistemic realizations;
% Na:     the scalar value of the number of aleatory realizations;
% model:  the SDF function-handle;
%
% output:
% output.samples:    the Na x Ne matrix of sample outputs;
% output.pbox:       the Na x 2 matrix of sample outputs for P-box;
% output.time:       the total time elapsed by the DLMC procedure;
%
%
%% Define the epistemic hyper-rectangle:
tic;
%input_samps = [unifrnd(bounds(1,1), bounds(1,2), Ne, 1), unifrnd(bounds(2,1), bounds(2,2), Ne, 1),...
%               unifrnd(bounds(3,1), bounds(3,2), Ne, 1), unifrnd(bounds(4,1), bounds(4,2), Ne, 1)]; % Ne x 4 matrix

%% Computation procedure:
samples = model(bounds); samples = sort(samples);

pbox = zeros(size(samples,1),2);
for i = 1:size(samples,1)
pbox(i,:) = [min(samples(i,:)), max(samples(i,:))]; 
end

timeDLMC = toc;
sprintf('Total time elapsed for the DLMC procedure is = %3f', timeDLMC)

%% Generate the outputs:
output.samples = samples;
output.pbox = pbox;
output.time = timeDLMC;

end

function [theta, acceptance_ratio]= post_staircaseRV(Nsample, Nale, bounds, alpha_cut)
% Return a column of samples from the posterior pdf
%

theta = zeros(Nsample, 4*Nale);
Ntotal = 0;
Naccepted = 0;

for ia = 1:Nale
    isample = 0;   % reset the counter
    
    while isample < Nsample
        tmp = zeros(Nsample, 4);
        tmp(:, 1) = alpha_cut(4*(ia - 1) + 1, 1) + diff(alpha_cut(4*(ia - 1) + 1, :))*rand(Nsample, 1);
        tmp(:, 2) = alpha_cut(4*(ia - 1) + 2, 1) + diff(alpha_cut(4*(ia - 1) + 2, :))*rand(Nsample, 1);
        tmp(:, 3) = alpha_cut(4*(ia - 1) + 3, 1) + diff(alpha_cut(4*(ia - 1) + 3, :))*rand(Nsample, 1);
        tmp(:, 4) = alpha_cut(4*(ia - 1) + 4, 1) + diff(alpha_cut(4*(ia - 1) + 4, :))*rand(Nsample, 1);
    
        % Do feasibility check on additional conditions
        Lfeasible = false(Nsample, 1);
        for i = 1:Nsample
            Lfeasible(i) = isfeasible(bounds, tmp(i, :));
        end
    
        Ntotal = Ntotal + Nsample;
        Naccepted = Naccepted + sum(Lfeasible);
        tmp(~Lfeasible, :) = [];
        theta(isample + 1:isample + sum(Lfeasible), 1 + 4*(ia - 1):4*ia) = tmp;
        isample = isample + sum(Lfeasible);
    end

    if size(theta, 1) > Nsample
        theta(Nsample + 1:end, :) = [];
    end
end

acceptance_ratio = Naccepted/Ntotal;
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

n_b = 25;   % n. of bins of staircase RVs
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
