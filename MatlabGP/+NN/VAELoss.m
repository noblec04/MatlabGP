classdef VAELoss

    properties
        beta=1;
    end

    methods

        function obj = VAELoss(beta)
            if nargin==1
                obj.beta = beta;
            end
        end

        function [e] = forward(obj,y,yp,zp)

            nz = size(zp,2)/2;

            mu = zp(:,1:nz);
            sig = exp(zp(:,nz+1:2*nz));

            DKL = 0.5*(sig + mu.^2 - 1 - log(sig));

            e = sum((y - yp).^2) + sum(sum(obj.beta*DKL));

        end
    end
end