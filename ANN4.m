% ANN-Based Simulation of Hepatitis B Model using RK4 and NSFD Data

clc; clear; close all;

% Parameters
lambda = 25; alpha = 0.000354; beta = 0.02; gamma = 0.00025;
sigma = 0.069; theta = 0.9; rho = 0.0028; mu = 0.01;
r = 3; phi = 0.08; u1 = 0.000835; u2 = 0.00995;

% Initial conditions
S0 = 5200; A0 = 200; C0 = 55; R0 = 25;
y0 = [S0; A0; C0; R0];

T = 1500; dt = 0.1;
t = 0:dt:T; N = length(t);

% Preallocate
S_rk = zeros(1,N); A_rk = S_rk; C_rk = S_rk; R_rk = S_rk;
S_nsfd = S_rk; A_nsfd = S_rk; C_nsfd = S_rk; R_nsfd = S_rk;
S_rk(1) = S0; A_rk(1) = A0; C_rk(1) = C0; R_rk(1) = R0;
S_nsfd(1) = S0; A_nsfd(1) = A0; C_nsfd(1) = C0; R_nsfd(1) = R0;

% RK4 Scheme
for i = 1:N-1
    S = S_rk(i); A = A_rk(i); C = C_rk(i); R = R_rk(i);
    f = @(S,A,C,R) [lambda*(1-alpha*C) - beta*(2*S*C)/(S+C) - gamma*A*S - u1*S - mu*S;
        gamma*A*S + beta*(2*S*C)/(S+C) - (sigma+theta+mu)*A;
        lambda*alpha*C - (mu+rho)*C + theta*A - (r*u2*C)/(1+phi^2);
        sigma*A + (r*u2*C)/(1+phi^2) + u1*S - mu*R];
    k1 = f(S,A,C,R);
    k2 = f(S+dt/2*k1(1), A+dt/2*k1(2), C+dt/2*k1(3), R+dt/2*k1(4));
    k3 = f(S+dt/2*k2(1), A+dt/2*k2(2), C+dt/2*k2(3), R+dt/2*k2(4));
    k4 = f(S+dt*k3(1), A+dt*k3(2), C+dt*k3(3), R+dt*k3(4));
    S_rk(i+1) = S + dt/6*(k1(1)+2*k2(1)+2*k3(1)+k4(1));
    A_rk(i+1) = A + dt/6*(k1(2)+2*k2(2)+2*k3(2)+k4(2));
    C_rk(i+1) = C + dt/6*(k1(3)+2*k2(3)+2*k3(3)+k4(3));
    R_rk(i+1) = R + dt/6*(k1(4)+2*k2(4)+2*k3(4)+k4(4));
end

% NSFD Scheme
phi_dt = @(h) (1 - exp(-h)) / h;
phi_h = phi_dt(dt);
for n = 1:N-1
    S = S_nsfd(n); A = A_nsfd(n); C = C_nsfd(n); R = R_nsfd(n);
    S_nsfd(n+1) = (S + phi_h*lambda*(1-alpha*C)) / ...
        (1 + phi_h*((2*beta*C)/(S+C) + gamma*A + u1 + mu));
    A_nsfd(n+1) = (A + phi_h*(beta*(2*S*C)/(S+C) + gamma*A*S)) / ...
        (1 + phi_h*(sigma + theta + mu));
    C_nsfd(n+1) = (C + phi_h*(lambda*alpha*C + theta*A_nsfd(n+1))) / ...
        (1 + phi_h*(rho + mu + (r*u2)/(1+phi^2)));
    R_nsfd(n+1) = (R + phi_h*(u1*S_nsfd(n+1) + sigma*A_nsfd(n+1) + ...
        (r*u2*C_nsfd(n+1))/(1+phi^2))) / (1 + phi_h*mu);
end

% ANN Training
inputs = t;
targets = [S_rk; A_rk; C_rk; R_rk; S_nsfd; A_nsfd; C_nsfd; R_nsfd];
net = feedforwardnet([20 15 10], 'trainlm');
net.trainParam.epochs = 1000; net.trainParam.goal = 1e-6;
[net,tr] = train(net, inputs, targets);
outputs = net(inputs);

% Plot RK4 vs ANN
figure('Name','RK4 vs ANN');
compartments = {'S','A','C','R'};
for i = 1:4
    subplot(2,2,i);
    plot(t, targets(i,:), 'b', t, outputs(i,:), 'r--');
    legend(['RK4-', compartments{i}],['ANN-', compartments{i}]);
    title(['RK4 vs ANN for ', compartments{i}]);
end

% Plot NSFD vs ANN
figure('Name','NSFD vs ANN');
for i = 5:8
    subplot(2,2,i-4);
    plot(t, targets(i,:), 'g', t, outputs(i,:), 'm--');
    legend(['NSFD-', compartments{i-4}],['ANN-', compartments{i-4}]);
    title(['NSFD vs ANN for ', compartments{i-4}]);
end

% Training-Validation performance
figure('Name','Training-Validation Performance');
plotperform(tr);

% --- Error Metrics ---
RMSE_rk = sqrt(mean((targets(1:4,:) - outputs(1:4,:)).^2, 2));
L2_rk = sqrt(sum((targets(1:4,:) - outputs(1:4,:)).^2, 2) / N);

RMSE_nsfd = sqrt(mean((targets(5:8,:) - outputs(5:8,:)).^2, 2));
L2_nsfd = sqrt(sum((targets(5:8,:) - outputs(5:8,:)).^2, 2) / N);

% Display Error Metrics
disp('RMSE for RK4 vs ANN:'); disp(RMSE_rk');
disp('L2 Norm for RK4 vs ANN:'); disp(L2_rk');
disp('RMSE for NSFD vs ANN:'); disp(RMSE_nsfd');
disp('L2 Norm for NSFD vs ANN:'); disp(L2_nsfd');
