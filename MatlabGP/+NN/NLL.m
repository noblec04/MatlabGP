classdef NLL

    methods

        function obj = NLL()


        end

        function [e] = forward(~,y,yp)

            mu = yp(:,1);
            sig = exp(yp(:,2));

            if size(y,1)==1
                e = sum(log(sig+eps) + ((y(1,:)'-mu).^2)./(sig.^2+eps));
            else
                e = sum(log(y(2,:)+eps) + ((y(1,:)'-mu).^2)./(y(2,:).^2+eps)) + sum((sig - y(2,:)).^2);
            end

        end
    end
end