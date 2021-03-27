% Author: Ricardo Baptista and Alessio Spantini and Youssef Marzouk
% Date:   May 2019
%
% See LICENSE.md for copyright information
%

function model = generate_data(model, J)

% Load model inputs
d         = model.d;
for_op    = model.for_op;
dt        = model.dt;
dt_iter   = model.dt_iter;
x0        = model.x0;

% Declare vectors to store state, observations and time
xt = zeros(d, J);

% Initialize xf and tf
xf = x0;

% Generate true data
for n=1:J

    % run dynamics and save results
	for i=1:dt_iter
		xf = for_op(xf, tf + dt*(i-1), dt);
	end

	% Save results in vector
	xt(:,n) = xf;

end

% Save data in model
model.xt     = xt;

end

% -- END OF FILE --
