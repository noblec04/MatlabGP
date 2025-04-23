close
clc

%%

% Parameters
N = 1000; % Number of points to sample
x_min = -4; x_max = 4;
y_min = -4; y_max = 4;
resolution = 100; % Resolution of the grid

% Create the grid
x = linspace(x_min, x_max, resolution);
y = linspace(y_min, y_max, resolution);
[X, Y] = meshgrid(x, y);

% Checkerboard pattern
length = 4;
checkerboard = mod(floor(X / (x_max - x_min) * length) + floor(Y / (y_max - y_min) * length), 2);

% Sample points in regions where checkerboard pattern is 1
sampled_points = [];
while size(sampled_points, 1) < N
    % Randomly sample a point within the x and y range
    x_sample = x_min + (x_max - x_min) * rand();
    y_sample = y_min + (y_max - y_min) * rand();
    
    % Determine the closest grid index
    i = floor((x_sample - x_min) / (x_max - x_min) * resolution) + 1;
    j = floor((y_sample - y_min) / (y_max - y_min) * resolution) + 1;
    
    % Check if the sampled point is in a region where checkerboard == 1
    if checkerboard(j, i) == 1
        sampled_points = [sampled_points; x_sample, y_sample];
    end
end

% Plot the checkerboard pattern
figure;
imagesc([x_min, x_max], [y_min, y_max], checkerboard);
colormap([0.5 0 0.5; 1 1 0]); % Purple and yellow
hold on;

% Plot sampled points
scatter(sampled_points(:, 1), sampled_points(:, 2), 'r', 'filled');
xlabel('X-axis');
ylabel('Y-axis');
hold off;

%%


% Noise and interpolation
t = 0.85;
noise = randn(N, 2);

% Interpolated points
interpolated_points = (1 - t) * noise + t * sampled_points;

% Plot
figure;
scatter(sampled_points(:, 1), sampled_points(:, 2), 'r', 'filled'); hold on;
scatter(noise(:, 1), noise(:, 2), 'b', 'filled');
scatter(interpolated_points(:, 1), interpolated_points(:, 2), 'g', 'filled');
hold off;

%%

% Training parameters
training_steps = 1000;
batch_size = 64;
losses = zeros(training_steps, 1);

for step = 1:training_steps
    % Sample data
    idx = randi(size(sampled_points, 1), batch_size, 1);
    x1 = sampled_points(idx, :);
    x0 = randn(size(x1));
    target = x1 - x0;
    t = rand(batch_size, 1);
    xt = (1 - t) .* x0 + t .* x1;

    % Forward pass
    pred = MLP(xt, t, 5, 512);

    % Loss calculation (mean squared error)
    loss = mean((target - pred).^2, 'all');
    losses(step) = loss;

    % Backpropagation and optimization would require a custom implementation
end

% Plot losses
figure;
plot(losses);
xlabel('Training Step');
ylabel('Loss');

%%

function output = MLP(input, t, layers, channels)
    % Example MLP function
    % input: Input data
    % t: Time embedding
    % layers: Number of layers
    % channels: Number of channels

    % Initialize weights (for simplicity, random initialization)
    weights = cell(layers, 1);
    for i = 1:layers
        weights{i} = randn(channels, channels);
    end

    % Input projection
    x = input * randn(size(input, 2), channels);

    % Time embedding projection
    t_emb = t * randn(size(t, 2), channels);
    x = x + t_emb;

    % Hidden layers
    for i = 1:layers
        x = max(0, x * weights{i}); % ReLU activation
    end

    % Output projection
    output = x * randn(channels, size(input, 2));
end