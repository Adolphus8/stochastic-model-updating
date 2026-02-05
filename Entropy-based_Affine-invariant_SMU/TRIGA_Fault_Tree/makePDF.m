function [f, map] = makePDF(post_samps, xin, Nbins)
%% Function-handle to construct PDF via Kernel Density Estimates:
%
% Inputs:
% post_samp: the N x dm vector of input samples;
% xin:       the j x n input vector of x values to compute the PDF;
% Nbins:     the no. of bins to construct the PDF;
%
% Output:
% f:         the 1 x n output vector of the PDF estimates;
% map:       the max aposteriori estimate for each parameter;
%% Define the parameters:
dm = size(post_samps,2); % The no. of distance metrics considered

for j = 1:dm
width = (max(post_samps(:,j)) - min(post_samps(:,j)))./Nbins;
pd = fitdist(post_samps(:,j), 'Kernel', 'Width', width);
f(j,:) = pdf(pd, xin(j,:))./max(pdf(pd, xin(j,:)));
end

map = zeros(dm,1);
for j = 1:dm
kdx = find(f(j,:) == max(f(j,:))); map(j) = xin(j,kdx);
end

end