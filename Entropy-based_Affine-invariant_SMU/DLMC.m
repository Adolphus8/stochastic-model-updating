function [output] = DLMC(bounds, model)
%% The Double-Loop Monte Carlo function:
%
% Inputs:
% bounds: bounds on the SDF;
% model:  the SDF function-handle;
%
% output:
% output.samples:    the Na x Ne matrix of sample outputs;
% output.pbox:       the Na x 2 matrix of sample outputs for P-box;
% output.time:       the total time elapsed by the DLMC procedure;
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;

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
