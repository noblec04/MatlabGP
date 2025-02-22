function B = ndgrid(varargin)
%NDGRID Rectangular grid in N-D space
%   [X1,X2,...,Xn] = NDGRID(x1,x2,...,xn) replicates the grid vectors
%   x1,x2,...,xn to produce the coordinates of a rectangular n-dimensional
%   grid (X1,X2,...,Xn). The i-th dimension of the output array Xi contains
%   copies of the grid vector xi. For example, each column of X1 is a copy
%   of x1, each row of X2 is a copy of x2 etc. All outputs Xi have size
%   length(x1)-by-length(x2)-by-...-by-length(xn).
%
%   [X1,X2,...,Xn] = NDGRID(x) is the same as to [X1,X2,...,Xn] =
%   NDGRID(x,x,...,x) and specifies a single grid vector x to use for all
%   n dimensions. The number of outputs determines the dimensionality n.
%
%   NDGRID outputs are typically used for the evaluation of functions of
%   multiple variables and for multidimensional interpolation.
%
%   MESHGRID and NDGRID are similar, but MESHGRID is restricted to 2-D and
%   3-D while NDGRID supports 1-D to N-D. In 2-D and 3-D the coordinates
%   returned by each function are the same. The difference is the shape of
%   their outputs. For grid vectors x, y, and z of length M, N, and P
%   respectively, NDGRID(x,y) outputs have size M-by-N while MESHGRID(x,y)
%   outputs have size N-by-M. Similarly, NDGRID(x,y,z) outputs have size
%   M-by-N-by-P while MESHGRID(x,y,z) outputs have size N-by-M-by-P.
%
%   Example: Evaluate the function x2*exp(-x1^2-x2^2-x^3) over the
%            range -2 <= x1 <= 2, -2 <= x2 <= 2, -2 <= x3 <= 2
%
%       [x1,x2,x3] = ndgrid(-2:.2:2, -2:.25:2, -2:.16:2);
%       z = x2 .* exp(-x1.^2 - x2.^2 - x3.^2);
%       slice(x2,x1,x3,z,[-1.2 .8 2],2,[-2 -.2])
%
%
%   Class support for inputs x1,x2,...,xn
%      float: double, single
%      integer: uint8, int8, uint16, int16, uint32, int32, uint64, int64
%
%   See also MESHGRID, SLICE, INTERPN, GRIDDEDINTERPOLANT.

%   Copyright 1984-2019 The MathWorks, Inc.

nout = nargin;

A = cell(1,max(nout,1));

j = 1:nout;
siz = cellfun(@numel,varargin);

for i=1:max(nout,1)
    x = full(varargin{j(i)});
    s = ones(1,nout);
    s(i) = numel(x);
    x = reshape(x,s);
    s = siz;
    s(i) = 1;
    A{i} = repmat(x,s);
end

B = [];

for i = 1:numel(A)
   B = [B A{i}(:)];
end

end