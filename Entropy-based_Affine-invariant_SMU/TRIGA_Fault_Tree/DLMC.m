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
Ne = 1000;
input_samps = [unifrnd(bounds(1,1), bounds(1,2), Ne, 1), unifrnd(bounds(2,1), bounds(2,2), Ne, 1)]; % Ne x 2 matrix

%% Computation procedure:
for i = 1:Ne
samples(:,i) = model(input_samps(i,:)); 
end
samples = sort(samples);

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