function [theta, acceptance_ratio]= prior_staircaseRV(Nsample, Nale, bounds, varargin)
% Return a column of samples from the prior pdf
%
% This function generates uniformly distributed samples for the epistemic
% parameters of the moments used to generate the staircase random
% variables. 
% The parameters are generated such that the sufficient conditions of a set
% of values for the first four central moments is satisfied.
% See Crespo et al. (2018)
%  
% This is obtained by doing rejection sampling.

theta = zeros(Nsample, 4*Nale);
Ntotal = 0;
Naccepted = 0;

if nargin >= 4
    alpha_cut = varargin{1};
end

for ia = 1:Nale
    isample = 0;   % reset the counter
    
    while isample < Nsample
        tmp = zeros(Nsample, 4);
        % g1, g2, g3: mean between upper and lower bounds
        if nargin >= 4
            tmp(:, 1) = alpha_cut(4*(ia - 1) + 1, 1)...
                + diff(alpha_cut(4*(ia - 1) + 1, :))*rand(Nsample, 1);
        else
            tmp(:, 1) = bounds(1) + diff(bounds)*rand(Nsample, 1);
        end
        
        % g4, g5: m2 between 0 and (b - a)^2/4
        if nargin >= 4
            tmp(:, 2) = alpha_cut(4*(ia - 1) + 2, 1)...
                + diff(alpha_cut(4*(ia - 1) + 2, :))*rand(Nsample, 1);
        else
            tmp(:, 2) = unifrnd(0, diff(bounds)^2/4, Nsample, 1);
        end
        
        % g9, g10: between specified bounds for m3
        if nargin >= 4
            tmp(:, 3) = alpha_cut(4*(ia - 1) + 3, 1)...
                + diff(alpha_cut(4*(ia - 1) + 3, :))*rand(Nsample, 1);
        else
            m3_lb = -diff(bounds)^3/(6*sqrt(3));
            m3_ub = diff(bounds)^3/(6*sqrt(3));
            tmp(:, 3) = unifrnd(m3_lb, m3_ub, Nsample, 1);
            tmp(:, 3) = tmp(:, 3)./(tmp(:, 2).^(3/2));
        end
    
        % g11, g12: m4 positive and obeying g12
        if nargin >= 4
            tmp(:, 4) = alpha_cut(4*(ia - 1) + 4, 1)...
                + diff(alpha_cut(4*(ia - 1) + 4, :))*rand(Nsample, 1);
        else
            tmp(:, 4) = unifrnd(0, diff(bounds)^4/12, Nsample, 1);
            tmp(:, 4) = tmp(:, 4)./(tmp(:, 2).^2);
        end
        
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