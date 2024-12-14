function [T,Y] = Simulator(F,X,optimizer,varargin)

for i = 1:X{1}.res

    for j = 1:numel(X)
        Xi(j,1) = X{j}.sample(1);
    end

    [tn,yn] = optimizer(F,Xi,varargin{:});

    for k = 1:length(tn)
        Y(:,k,i) = yn{k};
        T(k,i) = tn(k);
    end

end