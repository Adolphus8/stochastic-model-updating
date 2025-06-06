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