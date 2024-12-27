function A = ndgrid(B)
%NDGRID Rectangular grid in N-D space
%   [X[]] = NDGRID(X{}) replicates the grid vectors
%   x{1,..,N} to produce the coordinates of a rectangular n-dimensional
%   grid (X[]). 
%
%   NDGRID outputs are typically used for the evaluation of functions of
%   multiple variables and for multidimensional interpolation.
%
%       [X] = ndgrid({-2:.2:2, -2:.25:2, -2:.16:2});
%
% Based on buitin ndgrid function by MathWorks
%   Copyright 1984-2019 The MathWorks, Inc. 

siz = length(B);
j = 1:siz;

for i=1:siz
    x = full(B{j(i)});
    s = ones(1,nargout);
    s(i) = numel(x);
    x = reshape(x,s);
    s = siz;
    s(i) = 1;
    A(:,i) = repmat(x,s);
end

