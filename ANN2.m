clc; clear; close all;

%% PARAMETERS
lambda = 25; alpha = 0.000354; beta = 0.02; gamma = 0.00025;
sigma = 0.069; theta = 0.9; rho = 0.0028; mu = 0.01;
r = 3; phi = 0.08; u1 = 0.000835; u2 = 0.00995;

% Initial conditions
S0 = 5200; A0 = 200; C0 = 55; R0 = 25;
y0 = [S0; A0; C0; R0];

T = 1500; dt = 0.1; t = 0:dt:T; N = length(t);

%% TOGGLE: Choose Scheme ('rk4' or 'nsfd')
scheme = 'rk4';

%% SIMULATION

Y = zeros(4, N); Y(:,1) = y0;

if strcmp(scheme, 'rk4')
    for i = 1:N-1
        y = Y(:,i);
        k1 = modelODE(y);
        k2 = modelODE(y + 0.5*dt*k1);
        k3 = modelODE(y + 0.5*dt*k2);
        k4 = modelODE(y + dt*k3);
        Y(:,i+1) = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4);
    end
elseif strcmp(scheme, 'nsfd')
    for n = 1:N-1
        S = Y(1,n); A = Y(2,n); C = Y(3,n); R = Y(4,n);
        phi_h = phi_dt(dt);
        S_next = (S + phi_h * lambda * (1 - alpha * C)) / ...
                 (1 + phi_h * ((2 * beta * C) / (S + C) + gamma * A + u1 + mu));
        A_next = (A + phi_h * (beta * (2 * S * C) / (S + C) + gamma * A * S)) / ...
                 (1 + phi_h * (sigma + theta + mu));
        C_next = (C + phi_h * (lambda * alpha * C + theta * A_next)) / ...
                 (1 + phi_h * (rho + mu + (r * u2) / (1 + phi^2)));
        R_next = (R + phi_h * (u1 * S_next + sigma * A_next + (r * u2 * C_next) / (1 + phi^2))) / ...
                 (1 + phi_h * mu);
        Y(:,n+1) = [S_next; A_next; C_next; R_next];
    end
end

%% Prepare Training and Testing Data
split_ratio = 0.8;
split_idx = floor(split_ratio * N);
trainX = t(1:split_idx);  trainY = Y(:,1:split_idx);
testX = t(split_idx+1:end); testY = Y(:,split_idx+1:end);

%% Train Neural Network
net = feedforwardnet([20 15 10], 'trainlm');
net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-6;
net.divideParam.trainRatio = 100/100; % disable internal split
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 0;

[net, tr] = train(net, trainX, trainY);

%% Evaluate ANN
trainPred = net(trainX);
testPred = net(testX);

%% Compute RMSE
rmse_train = sqrt(mean((trainY - trainPred).^2, 2));
rmse_test = sqrt(mean((testY - testPred).^2, 2));

disp("RMSE on Train Set:"); disp(rmse_train');
disp("RMSE on Test Set:"); disp(rmse_test');

%% Plot ANN vs True for Test Set
figure;
compartments = {'S','A','C','R'};
for i = 1:4
    subplot(2,2,i)
    plot(testX, testY(i,:), 'b-', 'LineWidth', 2); hold on;
    plot(testX, testPred(i,:), 'r--', 'LineWidth', 2);
    xlabel('Time'); ylabel(compartments{i});
    title(['True vs ANN Prediction for ', compartments{i}]);
    legend('True', 'ANN'); grid on;
end

%% Plot Training Loss
figure;
plot(tr.perf, 'k', 'LineWidth', 2);
xlabel('Epoch'); ylabel('MSE Loss');
title('Training Performance (MSE Loss)'); grid on;

%% --- Functions ---
function dydt = modelODE(y)
    lambda = 25; alpha = 0.000354; beta = 0.02; gamma = 0.00025;
    sigma = 0.069; theta = 0.9; rho = 0.0028; mu = 0.01;
    r = 3; phi = 0.08; u1 = 0.000835; u2 = 0.00995;

    S = y(1); A = y(2); C = y(3); R = y(4);
    dS = lambda * (1 - alpha * C) - beta * (2 * S * C) / (S + C) - gamma * A * S - u1 * S - mu * S;
    dA = gamma * A * S + beta * (2 * S * C) / (S + C) - (sigma + mu + theta) * A;
    dC = lambda * alpha * C + theta * A - (mu + rho + (r * u2)/(1 + phi^2)) * C;
    dR = sigma * A + (r * u2 * C) / (1 + phi^2) + u1 * S - mu * R;
    dydt = [dS; dA; dC; dR];
end

function phi_val = phi_dt(dt)
    phi_val = dt / (1 + dt);  % Example nonstandard denominator function
end
