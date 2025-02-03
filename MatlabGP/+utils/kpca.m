function [E,score,v,mu] = kpca(X,m,kernel,precision)
% KM_KPCA_ICD performs kernel principal component analysis (KPCA) on a data
% set X, using Incomplete Cholesky Decomposition to approximate the large
% kernel matrix.
%
% Input:	- X: data matrix in column format (each data point is a row)
%			- m: the number of principal components to return. If m is 
%			smaller than 1, it is interpreted as the fraction of the signal
%			energy that is to be contained within the returned principal
%			components.
%			- ktype: string representing kernel type.
%			- kpar: vector containing the kernel parameters.
% Output:	- E: matrix containing the principal components.
%			- v: array containing the eigenvalues.
% USAGE: [E,v] = km_kpca_icd(X,m,ktype,kpar,precision)
%
% Author: Steven Van Vaerenbergh (steven.vanvaerenbergh at unican.es) 2010.
%
% This file is part of the Kernel Methods Toolbox for MATLAB.
% https://github.com/steven2358/kmbox

if nargin<5
	precision = 10^-6;
end
n = size(X,1);

mu = nanmean(X);

X = X - mu;

G = kernels.ICD(X',kernel,m,precision);

m1 = min(m,size(G,2));

R = G'*G;

[Er,Vr] = eig(R);

v = diag(Vr);

E = G*Er*diag(1./sqrt(v));

[v,ind] = sort(v,'descend');

E = E(:,ind(1:m1));	% principal components

v = v(1:m1);

for i=1:m1
	E(:,i) = E(:,i)/sqrt(n*v(i));	% normalization
end

score = X/E';

v = v/sum(v);