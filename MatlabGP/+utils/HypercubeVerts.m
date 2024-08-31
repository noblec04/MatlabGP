function [vertices] = HypercubeVerts(d)

%{
    Construct vector of hypercube vertices
%}

y = ones(1, d) * 2;
x = fliplr([1 cumprod(y)]);
n = x(1);
x = x(2:end);
vertices = ceil(repmat((1:n).', 1, d) ./ repmat(x, n, 1));
vertices = mod(vertices - 1, 2);

end