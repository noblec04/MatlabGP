function Aout = resize(A, n1, m1)
%INTERPOLATE_2D Performs 2D linear interpolation (Bilinear Interpolation)
%
%   OUTPUT_MATRIX = INTERPOLATE_2D(INPUT_MATRIX, N1, M1) resizes an N x M
%   matrix to an N1 x M1 matrix using bilinear interpolation.
%
%   Args:
%       input_matrix: The original N x M matrix (data grid).
%       n1: The desired number of output rows (N1).
%       m1: The desired number of output columns (M1).
%
%   Returns:
%       output_matrix: The new N1 x M1 interpolated matrix.

if isa(A,'AutoDiff')

    nz = size(getderivs(A),2);

    Av = getvalue(A);

    Aout.values = img.resize(Av,n1, m1);

    for i = 1:nz
        q = full(reshape(A.derivatives(:,i),size(Av)));
        p = img.resize(q,n1, m1);
        Aout.derivatives(:,i) = p(:);
    end

    Aout = AutoDiff(Aout);
    return
end


% Get the dimensions of the input matrix (N = rows, M = columns)
[n, m] = size(A);

% Initialize the output matrix with zeros
Aout = zeros(n1, m1);

% --- 1. Calculate Scaling Factors ---
% Scale factors map the range of steps in the output matrix (N1-1 steps)
% to the range of steps in the input matrix (N-1 steps).

% Scaling factor for rows (x-axis in source)
if n1 > 1
    sx = (n - 1) / (n1 - 1);
else
    % Handle case where output size is 1 (maps all to the first row/col)
    sx = 0;
end

% Scaling factor for columns (y-axis in source)
if m1 > 1
    sy = (m - 1) / (m1 - 1);
else
    sy = 0;
end

% Iterate over the output matrix coordinates (i for rows, j for columns)
for i = 1:n1
    for j = 1:m1

        % --- 2. Calculate corresponding floating-point coordinates (x, y) in the source matrix (1-based) ---
        % The formula maps output index (i-1) in range [0, N1-1] to source index range [0, N-1],
        % then converts back to 1-based by adding 1.
        x = 1 + (i - 1) * sx;
        y = 1 + (j - 1) * sy;

        % Determine the integer floor indices (Q11's coordinates)
        % These are the 1-based indices of the top-left neighbor
        x1 = floor(x);
        y1 = floor(y);

        % Clamp the indices: x2/y2 must not exceed the maximum index (n or m).
        % x2 and y2 are the 1-based indices of the bottom-right neighbor.
        x2 = min(x1 + 1, n);
        y2 = min(y1 + 1, m);

        % Get the values of the four surrounding points (Q11, Q21, Q12, Q22)
        Q11 = A(x1, y1); % Top-Left
        Q21 = A(x2, y1); % Top-Right (Same row index as Q11, next column index)
        Q12 = A(x1, y2); % Bottom-Left (Next row index as Q11, same column index)
        Q22 = A(x2, y2); % Bottom-Right

        % Calculate interpolation weights (the fractional part of x and y)
        dx = x - x1; % Weight for interpolation in the x (row) direction
        dy = y - y1; % Weight for interpolation in the y (column) direction

        % --- 3. Interpolate along the x-axis (Horizontal) ---
        % R1: Interpolated value at source row y1 (between Q11 and Q21)
        R1 = Q11 * (1 - dx) + Q21 * dx;

        % R2: Interpolated value at source row y2 (between Q12 and Q22)
        R2 = Q12 * (1 - dx) + Q22 * dx;

        % --- 4. Interpolate along the y-axis (Vertical) ---
        % P: Final interpolated value (between R1 and R2)
        P = R1 * (1 - dy) + R2 * dy;

        % Assign the final value to the output matrix
        Aout(i, j) = P;
    end
end
end


