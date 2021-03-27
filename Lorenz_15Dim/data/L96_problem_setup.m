% Author: Ricardo Baptista and Alessio Spantini and Youssef Marzouk
% Date:   May 2019
%
% See LICENSE.md for copyright information
%

%% ----------------------------
%% SETUP DA PROBLEM
%% ----------------------------

% Define number of variables
d = 15;

% Define number of steps
T = 11000;
T_SpinUp = 1000;

% Define time-stepping
dt = 0.01;
dt_iter_vect = 40;%20:20:60;

% define array to store auto-correlation
IACT_vs_dt = zeros(length(dt_iter_vect),1);

for i=1:length(dt_iter_vect)

    fprintf('Running L96 for Delta t=%d\n', dt_iter_vect(i))
    
	% set initial condition for data generation & spin-up
	m0 = zeros(d,1);
	C0 = eye(d);
	x0 = (m0 + sqrtm(C0)*randn(d,1))';
	
	% Setup forward operator
	F = 8;
	for_op = @(xt, t, dt) rk4(@(t,y) lorenz96(t,y,F), xt, t, dt);
	
	% define model
	model = struct;
	model.d       = d;
	model.dt      = dt;
	model.dt_iter = dt_iter_vect(i);
	model.m0      = m0;
	model.C0      = C0;
	model.x0      = x0;
	model.for_op  = for_op;
	
	% run dynamics and generate data
	model = generate_data(model, T);
	
	% remove spin-up samples
	data = model.xt(:,T_SpinUp:end);
	save(['L96_d' num2str(d) '_dt' num2str(dt_iter_vect(i))],'model','data');
	
    % compute integrated auto-correlation
    [~,~,~,IACT_vs_dt(i)] = UWerr_fft(data',[],[],[],[]);

	% plot auto-correlation between the chain elements
	figure('position',[0,0,1500,1500])
	for j=1:d
		subplot(4,5,j)
		autocorr(data(j,:))
	end
	print('-depsc',['autocorr_d' num2str(d) '_dt' num2str(dt_iter_vect(i))])

end

% plot integrated auto-correlation
figure
hold on
plot(dt_iter_vect*dt, IACT_vs_dt, '-o')
xlabel('Inter-observation time $\Delta t$')
ylabel('Integrated autocorrelation')
hold off
print('-depsc','IACT_vs_dt')

% -- END OF FILE --
