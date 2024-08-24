classdef NLL

    methods

        function obj = NLL()


        end

        function [e] = forward(~,y,yp)

            mu = yp(:,1);
            sig = exp(yp(:,2));

            e = sum(log(sig+eps) + ((y'-mu).^2)./(sig.^2+eps));

        end
    end
end