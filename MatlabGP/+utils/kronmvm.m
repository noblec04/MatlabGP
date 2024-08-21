% Perform a matrix vector multiplication b = A*x with a matrix A being a
% Kronecker product given by A = kron( kron(...,As{2}), As{1} ).
function b = kronmvm(As,x,transp)
if nargin>2 && ~isempty(transp) && transp   % transposition by transposing parts
  for i=1:numel(As)
      As{i} = As{i}';
  end
end
m = zeros(numel(As),1); n = zeros(numel(As),1);                  % extract sizes
for i=1:numel(n)
    [m(i),n(i)] = size(As{i});
end
d = size(x,2);
b = x;
for i=1:numel(n)                              % apply As{i} to the 2nd dimension
  sa = [prod(m(1:i-1)), n(i), prod(n(i+1:end))*d];                        % size
  a = reshape(permute(reshape(full(b),sa),[2,1,3]),n(i),[]);
  b = As{i}*a;
  b = permute(reshape(b,m(i),sa(1),sa(3)),[2,1,3]);
end
b = reshape(b,prod(m),d);                        % bring result in correct shape