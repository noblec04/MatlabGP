function [Xnew,newset] = discrete_max_min(X,n,ustart)

    % Generate discrete min max subsampling of a given sample set
    %(<a href="matlab:a=fileparts(which('CODES.install'));file=strcat(a,'/+doc/html/discrete_max_min.html');web(file);">HTML</a>)
    %
    % Syntax
    %   x=CODES.sampling.discrete_max_min(X,n,ustart) perform a subsampling
    %   of X at n points starting with ustart.
    %
    % Example
    %   x=lhsdesign(1000,2);
    %   xn = CODES.sampling.discrete_max_min(x,100);
    %   plot(x(:,1),x(:,2),'+')
    %   hold on
    %   plot(xn(:,1),xn(:,2),'o')
    %   - Generates a subsampling at 100 points
    %
    % Copyright 2013-2022 Computational Optimal Design of Engineering
    % Systems (CODES) laboratory

if nargin<3
ustart = 1; 
end

newset=[ustart];

for i=2:n
distnew=pdist2(X,X(newset,:));
[~,u]=max(min(distnew,[],2));
newset=[newset u];
end

Xnew = X(newset,:);

end

