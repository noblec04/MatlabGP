classdef NLL

    properties
        dist = 'normal'
    end

    methods

        function obj = NLL(dist)

            if nargin==0
                dist = 'normal';
            end
            obj.dist = dist;

        end

        function [e] = forward(obj,y,yp)
            switch obj.dist
                case 'normal'

                    mu = yp(:,1);
                    sig = exp(yp(:,2));

                    if size(y,1)==1
                        e = sum(log(sig+eps) + ((y(1,:)'-mu).^2)./(sig.^2+eps));
                    else
                        e = sum(log(y(2,:)+eps) + ((y(1,:)'-mu).^2)./(y(2,:).^2+eps)) + sum((sig - y(2,:)).^2);
                    end

                case 'gamma'
                    
                    alpha = exp(yp(:,1));
                    lambda = exp(yp(:,2));

                    e = sum(log(gamma(alpha)) - alpha.*log(lambda) - (alpha-1).*log(y(:,1)) + lambda.*y(:,1));


            end

        end
    end
end